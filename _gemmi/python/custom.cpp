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
        "Interpolates a list of points."
    );
}
