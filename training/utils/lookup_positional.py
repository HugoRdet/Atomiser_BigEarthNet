import torch
import pytorch_lightning as pl

class Lookup_encoding(pl.LightningModule):


    def __init__(self, modalities_config,bands_info):
        super().__init__()
        self.config= modalities_config
        self.bands_info=bands_info
        self.modalities=None
        self.table=None
        self.table_wave=None

        self.init_config()
        self.init_lookup_table()
        self.init_lookup_table_wave()

    def init_config(self):
        modalities=[]

        for tmp_modality in self.config["train"]:
            resolution=self.config["train"][tmp_modality]['resolution']
            size= self.config["train"][tmp_modality]['size']
            modalities.append((10.0/resolution,int(size*120)))

            
        
        for tmp_modality in self.config["test"]:
            resolution=self.config["test"][tmp_modality]['resolution']
            size= self.config["test"][tmp_modality]['size']
            modalities.append((10.0/resolution,int(size*120)))


        for tmp_modality in self.config["validation"]:
            resolution=self.config["validation"][tmp_modality]['resolution']
            size= self.config["validation"][tmp_modality]['size']
            modalities.append((10.0/resolution,int(size*120)))
        
        
        

        self.modalities=modalities
    
    def init_lookup_table(self):
        table=dict()
        idx_torch_array=0

        for couple in self.modalities:
            resolution,size=couple
            resolution=int(resolution*1000)
            table[(resolution,size)]=idx_torch_array
            idx_torch_array+=size
            
                   
        
        self.table=table
        
    def init_lookup_table_wave(self):
        table=dict()
        idx_torch_array=0
            
        for sat in self.bands_info:
            sat_content=self.bands_info[sat]
            for band in sat_content:
                
                band_content=sat_content[band]
                
                if not "bandwidth" in band_content or not "central_wavelength" in band_content:
                    continue
                
                bandwidth=band_content["bandwidth"]
                central_wavelength=band_content["central_wavelength"]
                table[(int(bandwidth),int(central_wavelength))]=idx_torch_array
                idx_torch_array+=1
                
            
                   
        
        self.table_wave=table

        
        


    



