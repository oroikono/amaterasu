# Copyright 2024 The CAM Lab at ETH Zurich.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
File contains the metadata information about the following datasets
The file_system gives information about where each file is stored

    Dataset:
    2D Shear Layer Problem:   ShearLayer2D      ConditionalShearLayer2D
    2D Cloud Shock:           CloudShock2D      ConditionalCloudShock2D
    3D Shear Layer:           ShearLayer3D      ConditionalShearLayer3D
    3D Taylor Green Vortex:   TaylorGreen3D     ConditionalTaylorGreen3D
    3D Nozzle:                Nozzle3D          ConditionalNozzle3D
"""

DIR_PATH_LOADER = '/cluster/work/math/camlab-data/synthetic'

################################# 2D Datasets #################################

ShearLayer2D_Metadata = {
    'dataset_name': 'ShearLayer2D',
    'file_name': 'sl2d.nc',
    'origin': DIR_PATH_LOADER
}

CloudShock2D_Metadata = {
    'dataset_name': 'CloudShock2D',
    'file_name': 'cloud_shock.nc',
    'origin': DIR_PATH_LOADER
}

RichtmyerMeshkov2D_Metadata = {
    'dataset_name': 'RichtmyerMeshkov',
    'file_name': 'CE-RM.nc',
    'origin': DIR_PATH_LOADER
} 



################################# 3D Datasets #################################

ShearLayer3D_Metadata = {
    'dataset_name': 'ShearLayer3D',
    'file_name': 'IEU_3D_CylindricalShearFlowLowRes.nc',
    'origin': DIR_PATH_LOADER
}

ShearLayerSpectral3D_Metadata = {
    'dataset_name': 'ShearLayerSpectral3D',
    'file_name': 'IEU_3D_CylindricalShearFlowLowRes.nc',
    'origin': DIR_PATH_LOADER,
    "resolver_path": "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/eul_ns3d_mix1/TURBO_MASK_scratch_Base_10ep_8gpus_bs3_4acc_10000/predictions_ns_shear3d_generated_data/ns_shear3d_pdegym_plus_pred_1500.nc"
}


TaylorGreen3D_Metadata = {
    'dataset_name': 'TaylorGreen3D',
    'file_name': '/cluster/work/math/camlab-data/synthetic/IEU_3D_TaylorGreenLowRes.nc',
    'origin': DIR_PATH_LOADER
}


Nozzle3D_Metadata = {
    'dataset_name': 'Nozzle3D',
    'file_name': 'nozzle3d.nc',
    # 'origin': '/cluster/work/math/camlab-data/data'
    'origin': DIR_PATH_LOADER
}

ATM_MSC_3D_moist_Metadata = {
    'dataset_name': 'ATM_MSC_3D_moist',
    'file_name': 'ATM-MSC_3D_moist.nc',
    'origin': DIR_PATH_LOADER
}

ATM_MSC_3D_dry_Metadata = {
    'dataset_name': 'ATM_MSC_3D_dry ',
    'file_name': 'ATM-CBL_3D_dry.nc',
    'origin': DIR_PATH_LOADER
}

################################# Conditional 2D #################################

ConditionalShearLayer2D_Metadata = {
    'dataset_name': 'ConditionalShearLayer2D',
    'file_name': 'macro_micro_id_2d.nc',
    'origin': DIR_PATH_LOADER,
}


ConditionalCloudShock2D_Metadata = {
    'dataset_name': 'ConditionalCloudShock2D',
    'file_name': 'micro_macro_cloudshock.nc',
    'origin': DIR_PATH_LOADER
}

################################# Conditional 3D #################################

ConditionalATM_MSC_3D_moist_Metadata = {
    'file_name': 'ATM-MSC_3D_moist_macro2.nc',
    'dataset_name': 'ConditionalATM_MSC_3D_moist',
    'origin': DIR_PATH_LOADER
}

ConditionalShearLayer3D_Metadata = {
    'file_name': 'IEU_3D_MacroMicroCylindricalShearFlow.nc',
    'dataset_name': 'ConditionalShearLayer3D',
    'origin': DIR_PATH_LOADER
}

ConditionalShearLayerSpectral3D_Metadata = {
    'file_name': 'IEU_3D_MacroMicroCylindricalShearFlow.nc',
    'dataset_name': 'ConditionalShearLayerSpectral3D',
    'origin': DIR_PATH_LOADER,
    "resolver_path": "/cluster/work/math/braonic/TrainedModels/OOD_Generalization/eul_ns3d_mix1/TURBO_MASK_scratch_Base_10ep_8gpus_bs3_4acc_10000/predictions_ns_shear3d_mm_test"

}

ConditionalTaylorGreen3D_Metadata = {
    'dataset_name': 'ConditionalTaylorGreen3D',
    'file_name': '/cluster/work/math/camlab-data/synthetic/IEU_3D_MacroMicroTaylorGreen.nc',
    'origin': DIR_PATH_LOADER
}


ConditionalNozzle3D_Metadata = {
    'dataset_name': 'ConditionalNozzle3D',
    'file_name': 'nozzle3d_micro.nc',
    'origin': DIR_PATH_LOADER
}


METADATA_CLASSES = {
    "ShearLayer2D"              :   ShearLayer2D_Metadata,
    "CloudShock2D"              :   CloudShock2D_Metadata,
    "RichtmyerMeshkov2D"        :   RichtmyerMeshkov2D_Metadata,

    "ShearLayer3D"              :   ShearLayer3D_Metadata,
    "ShearLayerSpectral3D"      :   ShearLayerSpectral3D_Metadata,
    "TaylorGreen3D"             :   TaylorGreen3D_Metadata,
    "Nozzle3D"                  :   Nozzle3D_Metadata,
    "ATM_MSC_3D_moist"          :   ATM_MSC_3D_moist_Metadata,
    "ATM_MSC_3D_dry"            :   ATM_MSC_3D_dry_Metadata,

    "ConditionalShearLayer2D"   :   ConditionalShearLayer2D_Metadata,
    "ConditionalCloudShock2D"   :   ConditionalCloudShock2D_Metadata,
    
    "ConditionalShearLayer3D"   :   ConditionalShearLayer3D_Metadata,
    "ConditionalShearLayerSpectral3D"   :   ConditionalShearLayerSpectral3D_Metadata,
    "ConditionalTaylorGreen3D"  :   ConditionalTaylorGreen3D_Metadata,
    "ConditionalNozzle3D"       :   ConditionalNozzle3D_Metadata,

    "ConditionalATM_MSC_3D_moist": ConditionalATM_MSC_3D_moist_Metadata
}