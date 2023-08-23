import gemmi

unit_cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
grid = gemmi.FloatGrid(100, 100, 100)
grid.fill(1.0)
grid.set_unit_cell(unit_cell)
grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")

box = gemmi.FractionalBox()
box.minimum = gemmi.Fractional(-0.25, 0.25, 0.25)
box.maximum = gemmi.Fractional(0.25, 0.75, 0.75)

m = gemmi.Ccp4Map()
m.grid = grid
m.update_ccp4_header(2, True)
m.setup()
m.set_extent(box)
m.write_ccp4_map("fractional.ccp4")
