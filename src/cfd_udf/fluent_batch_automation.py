from ansys.fluent.core import launch_fluent
import os

###############################################################################
# USER INPUTS
###############################################################################

CASE_FILE = "data_files_test\\br150\\bent_ratio_br150_velocity_v5.0.cas.h5"
DATA_FILE = "data_files_test\\br150\\bent_ratio_br150_velocity_v5.0.dat.h5"

UDF_SOURCE = r"one_udf_to_bring_them_together_in_darkness.c"

PARTICLE_MATERIAL = "sand"

PARTICLE_DENSITY = 2650.0      # kg/m3
PARTICLE_DIAMETER = 2.0e-4     # m
PARTICLE_MASS_FLOW = 1.0e-6    # kg/s
PARTICLE_TEMPERATURE = 300.0   # K

INJECTION_SURFACE = "inlet"

CSV_FILE = r"dpm_results.csv"

###############################################################################
# START FLUENT
###############################################################################

solver = launch_fluent(
    precision="double",
    processor_count=8,
    mode="solver"
)

tui = solver.tui

###############################################################################
# READ CASE/DATA
###############################################################################

tui.file.read_case(CASE_FILE)
tui.file.read_data(DATA_FILE)

###############################################################################
# ASSIGN 7 UDM SLOTS
###############################################################################

tui.define.user_defined.user_defined_memory("7")

###############################################################################
# COMPILE AND LOAD UDF
###############################################################################

udf_dir = os.path.dirname(UDF_SOURCE)
udf_name = os.path.basename(UDF_SOURCE)

tui.define.user_defined.compiled_functions(
    "compile",
    "dpm_udf",
    "yes",
    udf_dir,
    udf_name,
    "",
)

tui.define.user_defined.compiled_functions(
    "load",
    "dpm_udf"
)

###############################################################################
# ENABLE DPM
###############################################################################

dir(tui.define.models.dpm.injections)
dir(tui.define.models.dpm.interaction)
dir(tui.define.models.dpm.options)
dir(tui.define.models.dpm.numerics)
# tui.define.models.dpm("yes")


###############################################################################
# ENABLE EROSION / ACCRETION
###############################################################################

tui.define.models.dpm.interaction("yes")

tui.define.models.dpm.erosion_accretion("yes")

###############################################################################
# PARTICLE MATERIAL
###############################################################################

tui.define.materials.change_create(
    PARTICLE_MATERIAL
)

###############################################################################
# CREATE SURFACE INJECTION
###############################################################################

tui.define.injections.create(
    "surface",
    "inj-1"
)

###############################################################################
# INJECTION SETTINGS
###############################################################################

tui.define.injections.set_injection_properties(
    "inj-1",
    "surface-name",
    INJECTION_SURFACE
)

tui.define.injections.set_injection_properties(
    "inj-1",
    "material",
    PARTICLE_MATERIAL
)

tui.define.injections.set_injection_properties(
    "inj-1",
    "diameter",
    str(PARTICLE_DIAMETER)
)

tui.define.injections.set_injection_properties(
    "inj-1",
    "density",
    str(PARTICLE_DENSITY)
)

# tui.define.injections.set_injection_properties(
#     "inj-1",
#     "temperature",
#     str(PARTICLE_TEMPERATURE)
# )

tui.define.injections.set_injection_properties(
    "inj-1",
    "mass-flow-rate",
    str(PARTICLE_MASS_FLOW)
)

###############################################################################
# RANDOMIZE START LOCATIONS
###############################################################################

tui.define.injections.set_injection_properties(
    "inj-1",
    "random-surface-points",
    "yes"
)

###############################################################################
# TRACK 20000 STREAMS
###############################################################################

tui.define.injections.set_injection_properties(
    "inj-1",
    "number-of-particle-streams",
    "20000"
)

###############################################################################
# DRW STOCHASTIC TRACKING
###############################################################################

tui.define.models.dpm.stochastic_tracking(
    "yes"
)

tui.define.models.discrete_phase.number_of_tries(
    "5"
)

###############################################################################
# TRACK PARTICLES
###############################################################################

tui.solve.initialize.initialize_flow()

tui.solve.discrete_phase.iterate()

###############################################################################
# EXPORT EROSION + UDM VALUES
###############################################################################

tui.file.export.ascii(
    CSV_FILE,
    "wall-zones",
    "oka-erosion-rate",
    "udm-0",
    "udm-1",
    "udm-2",
    "udm-3",
    "udm-4",
    "udm-5",
    "udm-6"
)

###############################################################################
# SAVE RESULTS
###############################################################################

tui.file.write_case_data(
    r"dpm_completed.cas.h5"
)

print("DPM automation completed.")