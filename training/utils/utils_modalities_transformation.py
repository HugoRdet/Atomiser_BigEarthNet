import torch
from .utils_dataset import read_yaml,save_yaml
from .image_utils import *
from .files_utils import*
from math import pi
import einops 
import numpy as np

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    if num_bands==-1:
        scales = torch.linspace(1., max_freq , max_freq, device = device, dtype = dtype)
    else:
        scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)

    #scales shape: [num_bands]
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
    #scales shape: [len(orig_x.shape),]

    x = x * scales * 2*pi
        
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x


class modalities_transformations_config:

    def __init__(self,configs_dataset,path_imgs_config="./data/Tiny_BigEarthNet/",bands_infos="./data/bands_info/bands.yaml",name_config=""):

        self.name_config=name_config
        self.configs_dataset=read_yaml(configs_dataset)
        self.groups=self.configs_dataset["groups"]
        self.bands_infos=read_yaml(bands_infos)["bands_sen2_info"]


        self.path=path_imgs_config+"transformations"
        ensure_folder_exists(self.path)
        self.encoded_fourier_cc=dict()
        self.encoded_fourier_wavength=dict()

        self.gaussian_means=[]
        self.gaussian_stds=[]


        self.dico_group_channels=dict()
        if self.path[-1]=="\\":
            self.path=path_imgs_config[:-1]

    def get_band_identifier(self,channel_idx):
        for band_key in self.bands_infos:
            band=self.bands_infos[band_key]
           
            if band["idx"]==channel_idx:
                return band_key
        return None
            

    def get_band_infos(self,band_identifier):
        return self.bands_infos[band_identifier]

    
    def get_group(self,channel_idx):
        channel_name=self.get_band_identifier(channel_idx)
        for group in self.groups:
            if channel_name in self.groups[group]:
                return group

    def get_opposite_bands(self,tmp_group):
        group_bands=self.configs_dataset["groups"][tmp_group]
        res_bands=[]

        for band in list(self.bands_infos.keys()):
            if not band in group_bands:
                res_bands.append(band)
        return res_bands
            
            
    def get_channels_from_froup(self,group):
        if group in self.dico_group_channels:
            return self.dico_group_channels[group]
        group_bands=self.configs_dataset["groups"][group]
        res_idxs=[]
        for band in group_bands:
            res_idxs.append(self.get_band_infos(band)["idx"])
        
        self.dico_group_channels[group]=res_idxs
        return res_idxs

    def get_opposite_channels_from_froup(self,group):
        groupe_name=f"opposite_{group}"
        if groupe_name in self.dico_group_channels:
            return self.dico_group_channels[groupe_name]
        
        group_bands=self.get_opposite_bands(group)
        res_idxs=[]
        for band in group_bands:
            res_idxs.append(self.get_band_infos(band)["idx"])
        
        self.dico_group_channels[group]=res_idxs
        return res_idxs
            
    def get_random_attribute_id(self,L_attributes):
        if type(L_attributes)==list:
            rand_idx=int(torch.rand(())*len(L_attributes))
            return rand_idx
        if type(L_attributes)==dict:
            rand_idx=int(torch.rand(())*len(L_attributes.keys()))
            return list(L_attributes.keys())[rand_idx]
            
    def create_transform_image_dico(self,img_idx,mode,modality_folder=None):
        """create a transform yaml for each image
        """
        target_dico=dict()
        modalities=self.configs_dataset[mode]

        if modality_folder==None:
            modality_folder=mode
        
        modality_index=self.get_random_attribute_id(modalities)
        selected_modality=self.configs_dataset[mode][modality_index]
   
        for transfo_key in selected_modality.keys():
            if selected_modality[transfo_key]==None:
                continue
            
            transfo_val=selected_modality[transfo_key]

            if (type(transfo_val)==float and transfo_val==1.0) or transfo_val=="None":
                continue
            
            

            if transfo_key=="size":

                orig_img_size=self.configs_dataset["metadata"]["img_size"]
                new_size=None
                if type(transfo_val)==dict:
                    #infinite modalities
                    min_value=transfo_val["min"]
                    max_value=transfo_val["max"]
                    step=transfo_val["step"]

                    new_size=random_value_from_range(min_value, max_value, step)
                    new_size=int(orig_img_size*float(new_size))
                else:
                    new_size=int(orig_img_size*float(transfo_val))
                transfo_val=change_size_get_only_coordinates(orig_img_size,new_size)
            
            if transfo_key=="resolution":
                if type(transfo_val)==dict:
                    #infinite modalities
                    min_value=transfo_val["min"]
                    max_value=transfo_val["max"]
                    step=transfo_val["step"]

                    transfo_val=random_value_from_range(min_value, max_value, step)

            if transfo_key=="remove" or transfo_key=="keep":
                if transfo_val=="random":
                    transfo_val=self.get_random_attribute_id(self.groups)
                
            
            target_dico[transfo_key]=transfo_val

        folder_path=save_path=f"{self.path}/{modality_folder}/"

        if self.name_config!="":
            folder_path=save_path=f"{self.path}/{self.name_config}"
            ensure_folder_exists(folder_path)
            folder_path=save_path=f"{self.path}/{self.name_config}/{modality_folder}/"
        else:
            ensure_folder_exists(folder_path)



        save_path=f"{self.path}/{modality_folder}/{img_idx}_transfos_{mode}.yaml"
        if self.name_config!="":
            save_path=f"{self.path}/{self.name_config}/{modality_folder}/{img_idx}_transfos_{mode}.yaml"
        save_yaml(save_path,target_dico)

    def apply_transformations(self,img,mask,idx,mode="train",modality_mode=None):
        """
        apply transformations specified in {self.path}/{idx}_transfos.yaml file.
        This is the function you should call in the get_item
        """

        if modality_mode==None:
            modality_mode=mode


        file_path=f"{self.path}/{mode}/{idx}_transfos_{modality_mode}.yaml"
        if self.name_config!="":
            file_path=f"{self.path}/{self.name_config}/{mode}/{idx}_transfos_{modality_mode}.yaml"

        transfos=read_yaml(file_path)
     
        
        
        if "resolution" in transfos:
            new_resolution=int(img.shape[1]*float(transfos["resolution"]))
            img,mask=change_resolution(img=img,mask=mask,target_size=new_resolution)

        if "remove"in transfos:
            img,mask=remove_bands(img,mask,self.get_channels_from_froup(transfos["remove"]))

        if "size" in transfos:     
            img,mask=change_size(img,mask,transfos["size"])

        if "keep" in transfos:
            img,mask=remove_bands(img,mask,self.get_opposite_channels_from_froup(transfos["keep"]))

        

        
        return img,mask
    

    




        
     
      



        

    

    






    


    


        
        
    
    
