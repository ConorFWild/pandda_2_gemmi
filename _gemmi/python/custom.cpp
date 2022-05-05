// An empty file that makes easier to add custom functions.
//
#include <iostream>

#include "gemmi/ccp4.hpp"
#include "gemmi/gz.hpp"  // for MaybeGzipped
#include "gemmi/neighbor.hpp"
#include "gemmi/tostr.hpp"
#include "gemmi/fourier.hpp"  // for get_f_phi_on_grid

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


std::vec<float> interpolate_pos_array(
  Grid<float>& grid,
  py::array_t<float> pos_array,
  py::array_t<float> vals_array
){
  auto r_pos = pos_array.template mutable_unchecked<2>();
  auto r_val = vals_array.template mutable_unchecked<1>();
  std::vector<float> vals_vec;
  for (int i=0; i<r_pos.shape(0); i++){
    Position pos = Position(
      r_pos(i, 0),
      r_pos(i, 1),
      r_pos(i, 2)
      );
    auto val = grid.interpolate_value(pos);
    std::cout << val << "\n";
    r_val(i) = val;
    vals_vec[i] = val;
  }
  return vals_vec;

}

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



void add_custom(py::module& m) {
      m.def(
        "interpolate_points",
        &interpolate_points,
        "Interpolates a list of points and transforms."
    );
      m.def(
        "interpolate_pos_array",
        &interpolate_pos_array,
        py::arg().noconvert(),py::arg().noconvert(),py::arg().noconvert(),
        "Interpolates an array of points."
    );
          m.def(
        "interpolate_points_single",
        &interpolate_points_single,
        "Interpolates an list of points with a single transform."
    );
    
}
