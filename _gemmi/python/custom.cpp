// An empty file that makes easier to add custom functions.
//
#include <iostream>
#include <thread>

#include "gemmi/ccp4.hpp"
#include "gemmi/gz.hpp"  // for MaybeGzipped
#include "gemmi/neighbor.hpp"
#include "gemmi/tostr.hpp"
#include "gemmi/fourier.hpp"  // for get_f_phi_on_grid

#include <cassert>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "common.h"  // it includes <pybind11/pybind11.h>
namespace py = pybind11;

using namespace gemmi;


Grid<float> interpolate_points(
    Grid<float> moving_map,
    Grid<float> interpolated_map,
    std::vector<std::vector<int>> point_vec,
    std::vector<std::vector<double>> pos_vec,
    std::vector<Transform> transform_vec,
    std::vector<std::vector<double>> com_moving_vec,
    std::vector<std::vector<double>> com_reference_vec
    )
{
    for (std::size_t i=0; i < point_vec.size(); i++)
    {
        // Position
        std::vector<int> point = point_vec[i];

        Fractional fractional = Fractional(
            point[0] * (1.0 / interpolated_map.nu),
            point[1] * (1.0 / interpolated_map.nv),
            point[2] * (1.0 / interpolated_map.nw)
            );
        Position pos = interpolated_map.unit_cell.orthogonalize(fractional);
        Transform transform = transform_vec[i];
        std::vector<double> com_moving = com_moving_vec[i];
        std::vector<double> com_reference = com_reference_vec[i];

        //Subtract reference com
        pos.x -= com_reference[0];
        pos.y -= com_reference[1];
        pos.z -= com_reference[2];

        //transform
        Position pos_moving = Position(transform.apply(pos));

        // add moving com
        pos_moving.x += com_moving[0];
        pos_moving.y += com_moving[1];
        pos_moving.z += com_moving[2];

        // fractionalise
        Fractional pos_moving_fractional = moving_map.unit_cell.fractionalize(pos_moving);

        // interpolate
        float interpolated_value = moving_map.interpolate_value(pos_moving_fractional);

        // assign
        interpolated_map.set_value(
            point[0],
            point[1],
            point[2],
            interpolated_value
            );


    };

    return interpolated_map;

}

void interpolate_points_single(
    const Grid<float>& moving_map,
    Grid<float>& interpolated_map,
    const std::vector<std::vector<int>> point_vec,
    const std::vector<std::vector<double>> pos_vec,
    const Transform transform,
    const std::vector<double> com_moving,
    const std::vector<double> com_reference
    )
{
    for (std::size_t i=0; i < point_vec.size(); i++)
    {
        // Position
        std::vector<int> point = point_vec[i];

        Fractional fractional = Fractional(
            point[0] * (1.0 / interpolated_map.nu),
            point[1] * (1.0 / interpolated_map.nv),
            point[2] * (1.0 / interpolated_map.nw)
            );
          
        


        Position pos = interpolated_map.unit_cell.orthogonalize(fractional);
        // std::vector<float> pos_python = pos_vec[i];
        // Position pos = Position(pos_python[0], pos_python[1], pos_python[2]);

        // Transform transform = transform_vec[i];
        // std::vector<double> com_moving = com_moving_vec[i];
        // std::vector<double> com_reference = com_reference_vec[i];

        //Subtract reference com
        pos.x -= com_reference[0];
        pos.y -= com_reference[1];
        pos.z -= com_reference[2];

        //transform
        Position pos_moving = Position(transform.apply(pos));

        // add moving com
        pos_moving.x += com_moving[0];
        pos_moving.y += com_moving[1];
        pos_moving.z += com_moving[2];

        // fractionalise
        Fractional pos_moving_fractional = moving_map.unit_cell.fractionalize(pos_moving);

        // interpolate
        float interpolated_value = moving_map.interpolate_value(pos_moving_fractional);

        // assign
        interpolated_map.set_value(
            point[0],
            point[1],
            point[2],
            interpolated_value
            );


    };


}

void interpolate_points_single_array(
    const Grid<float>& moving_map,
    Grid<float>& interpolated_map,
     py::array_t<int> point_array,
     py::array_t<double> pos_array,
    const Transform transform,
    const std::vector<double> com_moving,
    const std::vector<double> com_reference
    )
{
    auto r_point = point_array.template mutable_unchecked<2>();
    auto r_pos = pos_array.template mutable_unchecked<2>();

    for (std::size_t i=0; i < r_point.shape(0); i++)
    {
        // Position
        Position pos = Position(
          r_pos(i,0),
          r_pos(i,1),
          r_pos(i,2)
        );

        //Subtract reference com
        pos.x -= com_reference[0];
        pos.y -= com_reference[1];
        pos.z -= com_reference[2];

        //transform
        Position pos_moving = Position(transform.apply(pos));

        // add moving com
        pos_moving.x += com_moving[0];
        pos_moving.y += com_moving[1];
        pos_moving.z += com_moving[2];

        // fractionalise
        Fractional pos_moving_fractional = moving_map.unit_cell.fractionalize(pos_moving);

        // interpolate
        float interpolated_value = moving_map.interpolate_value(pos_moving_fractional);

        // assign
        interpolated_map.set_value(
            r_point(i, 0),
            r_point(i, 1),
            r_point(i, 2),
            interpolated_value
            );


    };


}

void interpolate_points_multiple(
    const Grid<float>& moving_map,
    Grid<float>& interpolated_map,
     std::vector<py::array_t<int>> point_arr_vec,
     std::vector<py::array_t<double>> pos_arr_vec,
    std::vector<Transform> transform_vec,
    std::vector<std::vector<double>> com_moving_vec,
    std::vector<std::vector<double>> com_reference_vec
    )
{
    for (std::size_t i=0; i < point_arr_vec.size(); i++)
    {
      interpolate_points_single_array(
        moving_map,
        interpolated_map,
        point_arr_vec[i],
        pos_arr_vec[i],
        transform_vec[i],
        com_moving_vec[i],
        com_reference_vec[i]
      );

    };


}


void interpolate_points_multiple_parallel(
    const Grid<float>& moving_map,
    Grid<float>& interpolated_map,
     std::vector<py::array_t<int>> point_arr_vec,
     std::vector<py::array_t<double>> pos_arr_vec,
    const std::vector<Transform> transform_vec,
    const std::vector<std::vector<double>> com_moving_vec,
    const std::vector<std::vector<double>> com_reference_vec,
    const int num_threads
    )
{
    std::vector<std::thread> threads;

    // Get number of items to process with each thread
    int items_per_thread = (point_arr_vec.size() / num_threads) + 1;

    // Chunk and dispatch
    for (std::size_t thread_num=0; thread_num < num_threads; thread_num++)
    {
      std::vector<py::array_t<int>> point_arr_chunk;
      std::vector<py::array_t<double>> pos_arr_chunk;
      std::vector<Transform> transform_chunk;
      std::vector<std::vector<double>> com_moving_chunk;
      std::vector<std::vector<double>> com_reference_chunk;

      std::size_t initial = thread_num*items_per_thread;
      std::size_t point_arr_vec_size = point_arr_vec.size();
      std::size_t upper_bound = initial+items_per_thread;

      // Construct the chunks to process
      for (std::size_t k=initial; k < std::min(point_arr_vec_size, upper_bound) ; k++)
      {
        point_arr_chunk.push_back(point_arr_vec[k]);
        pos_arr_chunk.push_back(pos_arr_vec[k]);
        transform_chunk.push_back(transform_vec[k]);
        com_moving_chunk.push_back(com_moving_vec[k]);
        com_reference_chunk.push_back(com_reference_vec[k]);
      };

      // Dispatch a thread on the chunks
      threads.push_back(
        std::thread(
          interpolate_points_multiple,
          std::ref(moving_map),
          std::ref(interpolated_map),
          point_arr_chunk,
          pos_arr_chunk,
          transform_chunk,
          com_moving_chunk,
          com_reference_chunk
        )
      );

    };

    for (std::size_t k=0; k < threads.size(); k++) {
      threads[k].join();
    };



}




// std::vector<float> interpolate_pos_array(
//   Grid<float>& grid,
//   py::array_t<float> pos_array,
//   py::array_t<float> vals_array
// ){
//   auto r_pos = pos_array.template mutable_unchecked<2>();
//   auto r_val = vals_array.template mutable_unchecked<1>();
//   // std::vector<float> vals_vec;
//   for (int i=0; i<r_pos.shape(0); i++){
//     Position pos = Position(
//       r_pos(i, 0),
//       r_pos(i, 1),
//       r_pos(i, 2)
//       );
//     auto val = grid.interpolate_value(pos);
//     // std::cout << val << "\n";
//     r_val(i) = val;
//     // vals_vec.push_back(val);
//   }
//   return vals_vec;

// }

// int num_atoms(Structure structure){
//   int n = 0;

//   for (gemmi::Model& model : structure.models)
//       for (gemmi::Chain& chain : model.chains)
//         for (gemmi::Residue& res : chain.residues)
//           for (gemmi::Atom& atom : res.atoms)
//             if (atom.name != "H") {
//               n = n+1;            
//               }
        

//   return n;
// }

// Structure transform_structure(
//   Structure& structure, 
//     std::vector<float>& translation,
//   std::vector<std::vector<float>>& rotation,
//   )
// {
//   Structure structure_copy = new Structure(structure);

//   std::vector<float> structure_mean = get_structure_mean(structure_copy);



//   return structure_copy;
// }

// std::vector<float> transform_and_interpolate(
//   Structure& structure,
//   Grid<float>& xmap,
//   std::vector<float>& translation,
//   std::vector<std::vector<float>>& rotation,
// )
// {
//   Structure structure_copy = transform_structure(structure, translation, rotation);

//   int n = num_atoms(structure_copy);

//   std::vector<float> scores;

//   for (gemmi::Model& model : structure_copy.models)
//       for (gemmi::Chain& chain : model.chains)
//         for (gemmi::Residue& res : chain.residues)
//           for (gemmi::Atom& atom : res.atoms)
//             if (atom.name != "H") {
//               float score = xmap.interpolate_value(atom.pos); 
//               scores.push_back(score);   
//               }

//   return scores;

// }
void interpolate_pos_array(
   const Grid<float>& grid,
   py::array_t<float> pos_array,
   py::array_t<float> vals_array
)
{
  auto r_pos = pos_array.template mutable_unchecked<2>();
  auto r_val = vals_array.template mutable_unchecked<1>();
  for (int i=0; i<r_pos.shape(0); i++){
    Position pos = Position(
      r_pos(i, 0),
      r_pos(i, 1),
      r_pos(i, 2)
      );
    auto val = grid.interpolate_value(pos);
    r_val(i) = val;
}
}

void add_custom(py::module& m) {
      m.def(
        "interpolate_points",
        &interpolate_points,
        "Interpolates a list of points and transforms."
    );
      m.def(
        "interpolate_points_multiple",
        &interpolate_points_multiple,
        "Interpolates a list of points and transforms."
    );
          m.def(
        "interpolate_points_multiple_parallel",
        &interpolate_points_multiple_parallel,
        "Interpolates a list of points and transforms."
    );

    //   m.def(
    //     "interpolate_pos_array",
    //     &interpolate_pos_array,
    //     py::arg("grid"), py::arg("pos_array").noconvert(), py::arg("vals_array").noconvert(),
    //     "Interpolates an array of points."
    // );
//          m.def(
//        "interpolate_pos_array",
//        [](
//          const Grid<float>& grid,
//          py::array_t<float> pos_array,
//          py::array_t<float> vals_array
//        ){
//          auto r_pos = pos_array.template mutable_unchecked<2>();
//          auto r_val = vals_array.template mutable_unchecked<1>();
//          // std::vector<float> vals_vec;
//          for (int i=0; i<r_pos.shape(0); i++){
//            Position pos = Position(
//              r_pos(i, 0),
//              r_pos(i, 1),
//              r_pos(i, 2)
//              );
//            auto val = grid.interpolate_value(pos);
//            // std::cout << val << "\n";
//            r_val(i) = val;
//            // vals_vec.push_back(val);
//          }
//        // return vals_vec;
//        },
//        py::arg("grid"), py::arg("pos_array").noconvert(), py::arg("vals_array").noconvert(),
//        "Interpolates an array of points."
//    );
        m.def(
        "interpolate_pos_array",
        &interpolate_pos_array,
        "Interpolates an array of points."
    );
          m.def(
        "interpolate_points_single",
        &interpolate_points_single,
        "Interpolates an list of points with a single transform."
    );
    
}
