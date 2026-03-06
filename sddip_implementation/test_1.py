from msppy.utils.examples import construct_nvidi
from msppy.solver import SDDiP, Extensive
nvidi = construct_nvidi()
nvidi_ext = Extensive(nvidi)
nvidi_ext.solve(outputFlag=1)
nvidi_ext.first_stage_solution
nvidi_sddip = SDDiP(nvidi)
nvidi_sddip.solve(max_iterations=10, cuts=['B','SB','LG'])
nvidi_sddip.db[-1]
nvidi_sddip.first_stage_solution