
import os
from sgf_reader import SGFReader

class data_loader():


    non_adv_dir_path = 'Data/Non-Adv_Policies'

    adv_dir_path = 'Data/Adversarial_Policies'

    non_adv_matchups = [['base-adv', 'may23-vic'],['base-adv', 'dec23'],['base-adv', 'may24'],['base-adv','v9'],['base-adv', 'ViT-Vic'],['large','may24'],
                        ['cont','may24']]

    cursed_files = ['.DS_Store', 'game.dat']

    name_mapping = {'a9' : ['r9-v600'], 
                    'atari' : ['attack-h9-s564026112-v600-AMCTS', 'attack-h9-s564026112-v128-AMCTS', 'attack-h9-s564026112-v32-AMCTS', 
                               'attack-h9-s564026112-v48-AMCTS', 'attack-h9-s564026112-v24-AMCTS', 'attack-h9-s564026112-v1-AMCTS', 'attack-h9-s564026112-v16-AMCTS', 
                               'attack-h9-s564026112-v256-AMCTS', 'attack-h9-s564026112-v64-AMCTS', 'attack-h9-s564026112-v4-AMCTS','attack-h9-s564026112-v32-AMCTS', 
                               'attack-h9-s564026112-v48-AMCTS', 'attack-h9-s564026112-v24-AMCTS', 'attack-h9-s564026112-v1024-AMCTS', 'attack-h9-s564026112-v2048-AMCTS', 
                               'attack-h9-s564026112-v2-AMCTS', 'attack-h9-s564026112-v4096-AMCTS', 'attack-h9-s564026112-v8-AMCTS'],
                    'base-adv' : ['cyclic-v600', 'adv-s545065216-v600', 'r0-v600', 'adv-s545065216-v600-AMCTS-S++'],
                    'cont' : ['cont-s630m'],
                    'gift' : ['attack-b18-s651m-v256', 'attack-b18-s651m', 'attack-b18-s651m-v16', 'attack-b18-s651m-v64', 'attack-b18-s651m-v1', 'attack-b18-s651m-v4',
                              'attack-b18-s651m-v2', 'attack-b18-s651m-v128', 'attack-b18-s651m-v8', 'attack-b18-s651m-v2048', 'attack-b18-s651m-v32', 
                              'attack-b18-s651m-v4096', 'attack-b18-s651m-v1024'],
                    'large' : ['large-s216m'],
                    'stall' : ['attack-ft-h9-s97114624-v16-AMCTS', 'attack-ft-h9-s97114624-v32-AMCTS', 'attack-ft-h9-s97114624-v8-AMCTS', 'attack-ft-h9-s97114624-v2048-AMCTS',
                                'attack-ft-h9-s97114624-v1-AMCTS', 'attack-ft-h9-s97114624-v4096-AMCTS', 'attack-ft-h9-s97114624-v128-AMCTS', 'attack-ft-h9-s97114624-v2-AMCTS',
                                'attack-ft-h9-s97114624-v1024-AMCTS', 'attack-ft-h9-s97114624-v256-AMCTS', 'attack-ft-h9-s97114624-v4-AMCTS', 
                                'attack-ft-h9-s97114624-v64-AMCTS','attack-ft-h9-s97114624-v600-AMCTS'],
                    'ViT-adv' : ['attack-vit-s326m'],
                    'v9' : ['h9-v512', 'h9-v1024', 'h9-v64','h9-v8192', 'h9-v2048', 'h9-v4096', 'h9-v2', 'h9-v4', 'h9-v256', 'h9-v1', 'h9-v8', 'h9-v128', 'h9-v16', 'h9-v32',
                            'h9-v16384', 'h9-v32768'],
                    'dec23' : ['b18-s8527m-v512', 'b18-s8527m-v2048', 'b18-s8527m-v8', 'b18-s8527m-v1024', 'b18-s8527m-v4096', 'b18-s8527m-v8192','b18-s8527m-v2', 
                               'b18-s8527m-v32', 'b18-s8527m-v128', 'b18-s8527m-v64', 'b18-s8527m-v1','b18-s8527m-v2', 'b18-s8527m-v32', 'b18-s8527m-v128', 'b18-s8527m-v64', 
                               'b18-s8527m-v1','b18-s8527m-v4', 'b18-s8527m-v16', 'b18-s8527m-v256'],
                    'may23-vic' : ['b60-s7702m-v4096'],
                    'may24' : ['b18-s9997m-v256', 'b18-s9997m-v4096', 'b18-s9997m-v8192','b18-s9997m-v16384', 'b18-s9997m-v2048', 'b18-s9997m-v8', 'b18-s9997m-v32', 
                               'b18-s9997m-v2', 'b18-s9997m-v128', 'b18-s9997m-v4', 'b18-s9997m-v1024', 'b18-s9997m-v64', 'b18-s9997m-v16','b18-s9997m-v512', 'b18-s9997m-v1',
                               'b18-s9997m-v32768', 'b18-s9997m-v65536'],
                    'base-vic' : ['cp505-v1000000', 'cp505h-v2','cp505h-v16', 'cp505h-v128', 'cp505h-v4096', 'cp505h-v8192', 'cp505h-v1', 'cp505h-v1024', 'cp505h-v4', 
                                  'cp505h-v2048', 'cp505h-v32', 'cp505h-v32768', 'cp505h-v256', 'cp505h-v512', 'cp505h-v16384', 'cp505h-v64', 'cp505h-v8'],
                    'ViT-Vic' : ['vit-b16-s650m-v1024', 'vit-v1024', 'vit-b16-s650m-v128', 'vit-b16-s650m-v4', 'vit-b16-s650m-v64', 'vit-b16-s650m-v16', 
                                 'vit-b16-s650m-v256', 'vit-b16-s650m-v8', 'vit-b16-s650m-v2', 'vit-b16-s650m-v32', 'vit-b16-s650m-v1', 'vit-b16-s650m-v512', 
                                 'vit-v512', 'vit-v64', 'vit-v8', 'vit-v2048', 'vit-v8192', 'vit-v65536', 'vit-v128', 'vit-v4096', 'vit-v256', 'vit-v32', 'vit-v1', 
                                 'vit-v16', 'vit-v4', 'vit-v2']
                    }

    def load_data(self):

        data = ['black name','white name','black model', 'white model', 'adversarial', 'attack type', 'moves', 'file path']

        self.load_adv_model_data(self)




    def load_non_adv_model_data(self):
        
        directory = os.fsdecode(self.non_adv_dir_path)

        non_adv_data = []

        for sub_folder in os.listdir(directory):
            sub_folder_path = self.non_adv_dir_path +'/'+ os.fsdecode(sub_folder)
            if not os.fsdecode(sub_folder) in self.cursed_files:
                for file in os.listdir(sub_folder_path):
                    if not os.fsdecode(file) in self.cursed_files:
                        file_path = sub_folder_path + '/' + os.fsdecode(file)
                        game_list = SGFReader.read_file(SGFReader, file_path)
                        for game in game_list:
                            game_info = ['NA','NA',game['pla'], game['opp'], '0', 'NA', game['moves'], file_path]
                            non_adv_data.append(game_info)

        return(non_adv_data)


    def load_adv_model_data(self):
        
        directory = os.fsencode(self.adv_dir_path)

        adv_data = []

        for subfolder in os.listdir(directory):
            if not os.fsdecode(subfolder) in self.cursed_files:
                sub_folder_path = self.adv_dir_path + '/' + os.fsdecode(subfolder)
                for file in os.listdir(sub_folder_path):
                    if not os.fsdecode(file) in self.cursed_files:
                        file_path = sub_folder_path + '/' + os.fsdecode(file)
                        games_list = SGFReader.read_file(SGFReader, file_path)
                        for game in games_list:

                            black_name = 'NA'
                            white_name = 'NA'
                            black_model = game['pla']
                            white_model = game['opp']

                            for name in self.name_mapping.keys():
                                if black_model in self.name_mapping[name]:
                                    black_name = name
                                elif white_model in self.name_mapping[name]:
                                    white_name = name
                            
                            adversarial = 1

                            if (black_name in self.non_adv_matchups) and (white_name in self.non_adv_matchups):
                                adversarial = 0
                            elif (black_name in ['cont','dec23']) and (['cont','dec23']):
                                if (black_name != white_name):
                                    if (black_model in ['b18-s8527m-v4096', 'b18-s8527m-v8192']) or (white_model in 'b18-s8527m-v4096', 'b18-s8527m-v8192'):
                                        adversarial = 0
                                else:
                                    adversarial = 0
                            
                            if (black_name  == 'gift') or (white_name == 'gift'):
                                attack_type = 'non-cyclic'
                            else:
                                attack_type = 'cyclic'

                            game_info = [black_name, white_name, black_model, white_model, adversarial, attack_type, game['moves'], file_path]

                            adv_data.append(game_info)



    
