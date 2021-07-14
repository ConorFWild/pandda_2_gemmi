import gemmi

unit_cell = gemmi.UnitCell(10, 10, 10, 90, 90, 90)
grid = gemmi.FloatGrid(100, 100, 100)
grid.fill(1.0)
grid.set_unit_cell(unit_cell)
grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
#
# box = gemmi.PositionBox()
# box.minimum = gemmi.Position(-2.0, 2.0, 2.5)
# box.maximum = gemmi.Position(2.5, 7.5, 7.5)
#
# ccp4 = gemmi.Ccp4Map()
# ccp4.update_ccp4_header(2, True)
# ccp4.grid = grid
# ccp4.setup()
# ccp4.set_extent(box)
# ccp4.write_ccp4_map("position.ccp4")

box = gemmi.FractionalBox()
box.minimum = gemmi.Fractional(-0.25, 0.25, 0.25)
box.maximum = gemmi.Fractional(0.25, 0.75, 0.75)

ccp4 = gemmi.Ccp4Map()
ccp4.update_ccp4_header(2, True)
ccp4.grid = grid
ccp4.get_extent()
ccp4.setup()
ccp4.set_extent(box)
ccp4.write_ccp4_map("fractional.ccp4")
