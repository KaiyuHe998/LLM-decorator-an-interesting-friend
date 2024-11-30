from accelerate import hooks
from collections import deque
from sklearn.decomposition import PCA
import einops
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import *
from utils import honesty_function_dataset # TODO, define our own dataset layout
import utils
from itertools import islice
import numpy as np
from tqdm.notebook import tqdm

class rep_layer_hook(hooks.ModelHook): # this is a hook add to layers
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module_path_name = kwargs['module_path_name']
        self.mode = 'train' 
        self.direction_name = None
        self.control_intensity = -6
        # train: in train mode, hook store all information flow over this hook
        # control: in control mode, hook do not store informaiton and will adjust model forward function
        # freeze: Do nothing
        self.control_directions = {}
        self.signs = {}
        #self.freeze_cache = []
        self.post_hook_cache = []# {input_information: hiddenstate} all hiddenstate are saved here
        
    def pre_forward(self, module, *args, **kwargs):
        self.saved_args = args
        self.saved_kwargs = kwargs
        return args, kwargs

    def post_forward(self, module, output):
        # Currently do nothing about post_forward
        if self.mode == 'train':
            assert len(output) == 1, f'''output have multipler terms: {output}'''
            self.post_hook_cache.append(output[0][:,-1,:].detach().cpu()) #[batch, token_position, hidden_dim]
            return output
        elif self.mode == 'control':
            input_hiddenstates = output[0]
            norm_pre = torch.norm(input_hiddenstates, dim=-1, keepdim=True)
            assert len(self.control_directions) > 0 and len(self.signs) > 0, f'''direction and sign is none, call calculate_direction after gather enough data'''
            assert self.direction_name is not None, f'''You need to specify a generate direction name!'''
            control_vector = torch.tensor(self.control_intensity * self.control_directions[self.direction_name] * self.signs[self.direction_name]).to(input_hiddenstates.device).half()
            control_vector = control_vector.squeeze()
            control_vector = control_vector.reshape(1,1,-1)
            control_vector = control_vector.to(control_vector.device)
            control_vector.to(input_hiddenstates.dtype)
            control_vector = control_vector.expand(input_hiddenstates.shape[0],input_hiddenstates.shape[1],-1)
            attention_mask = self.saved_kwargs.get('attention_mask', None)
            if attention_mask is not None:
                print('yeah!!!!!')
                control_vector = control_vector * attention_mask
            else:
                control_vector = control_vector
            # Change the shape of control vector to match the input
            assert control_vector.shape == input_hiddenstates.shape, f'''shape between control vector and input hiddenstate are not the same.'''
            input_hiddenstates = input_hiddenstates + control_vector
            # update output
            if isinstance(output, tuple):
                # If output is a tuple, replace the first element
                output = (input_hiddenstates,) + output[1:]
            else:
                # If output is a tensor, simply assign the modified hidden states
                output = input_hiddenstates
            # Return the modified output
            return output
        elif self.mode == 'freeze':
            #self.freeze_cache.append(output) # do not store the activations saving the spaces
            return output
        else:
            raise NotImplementedError
        return output

    def get_signs(self, labels, direction_name:Union[str, tuple]): # direction_name is used when training should align with data
        """Given labels for the training data hidden_states, determine whether the
        negative or positive direction corresponds to low/high concept 
        (and return corresponding signs -1 or 1 for each layer and component index)
        
        NOTE: This method assumes that there are 2 entries in hidden_states per label, 
        aka len(hidden_states) == 2 * len(labels). For example, if 
        n_difference=1, then hidden_states here should be the raw hidden states
        rather than the relative (i.e. the differences between pairs of examples).

        Args:
            hidden_states: Hidden states of the model on the training data (current module for this hook)
            labels: Labels for the training data

        Returns:
            signs: A sign for current hook
        PCA is an unsupervised method, labels here are only used to find the sign for founded direction
        """        
        hidden_states = torch.cat(self.post_hook_cache)
        assert hidden_states.shape[0] == 2 * len(labels), f"Shape mismatch between hidden states ({hidden_states.shape[0]}) and labels ({len(labels)})"
        # get the signs for each component
        component_signs = np.zeros(self.n_components)
        for component_index in range(self.n_components):
            transformed_hidden_states = utils.project_onto_direction(hidden_states, self.control_directions[direction_name][component_index])
            transformed_hidden_states = transformed_hidden_states.cpu().numpy()
            pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in labels[:i]), sum(len(c) for c in labels[:i+1]))) for i in range(len(labels))]

            # We do elements instead of argmin/max because sometimes we pad random choices in training
            pca_outputs_min = np.mean([o[labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
            pca_outputs_max = np.mean([o[labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])
            
            component_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
            if component_signs[component_index] == 0:
                component_signs[component_index] = 1 # default to positive in case of tie
        self.signs[direction_name] = component_signs
        
    def get_rep_directions(self, direction_name:Union[str, tuple], n_components = 1):
        """Get PCA components for each layer"""
        hidden_states = torch.cat(self.post_hook_cache)
        hidden_states = hidden_states[::2] - hidden_states[1::2] # !!! important read the relative concept direction from positive and negative samples
        H_train = hidden_states
        H_train_mean = H_train.mean(axis=0, keepdims=True)
        self.H_train_mean = H_train_mean
        H_train = utils.recenter(H_train, mean=H_train_mean).cpu()
        H_train = np.vstack(H_train)
        pca_model = PCA(n_components=1, whiten=False).fit(H_train) # by default set n_components to 1

        assert direction_name not in self.control_directions, f'''{direction_name} already exist in current direction dict'''
        self.control_directions[direction_name] = pca_model.components_ # shape (n_components, n_features)
        self.n_components = pca_model.n_components_

    def calculate_direction(self, labels, tuple_direction_name:tuple):
        '''Generate controll directions based on current method and saved '''
        '''Currently using PCA for direction extraction'''
        assert tuple_direction_name not in self.control_directions, f'''Currently you are training {tuple_direction_name} direction for hooks, but this is already an existing direction'''
        # train key (direction, neg)

        self.get_rep_directions(tuple_direction_name)
        self.get_signs(labels, tuple_direction_name)
        


class hooked_transformer:
    def __init__(self, 
                 model_name:str,
                 hg_token:str = None,):
        self.model_name = model_name
        self.hg_token = hg_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          do_sample = False,
                                                          temperature = 0,
                                                          return_dict_in_generate = True, 
                                                          output_hidden_states=True, 
                                                          device_map="auto",
                                                          token = self.hg_token).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token = self.hg_token,padding_side="left",)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.print_store_module_info()
        self.hook_module_list = []
        self.supported_hook_dict ={'rep_layer_hook':rep_layer_hook}
        self.hiddenstate_label = []
        self.mode = 'train'
        
    def __call__(self, *args, **kwargs):
        if self.mode == 'train':
            assert 'input_data_label' in kwargs, f'''Current model mode is set to train, you need to input data label for training'''
            self.hiddenstate_label.extend(kwargs['input_data_label'])
            del kwargs['input_data_label']

        return self.model.__call__(*args, **kwargs)
            
        
    def print_store_module_info(self):
        prefix = ''
        self.all_modules = {}
        print('----------------------------------------------------------------------------------------')
        print("This function print all the modules in this model,")
        print("since different model will have different architecture and different module name......")
        print("please select the module you want to hook manually after check the forward function and the layout of the modules")
        print('----------------------------------------------------------------------------------------')
        stack = deque([(self.model, prefix)])
        while stack:
            current_module, current_prefix = stack.pop()
            children = list(current_module.named_children())
            for name, sub_module in reversed(children):
                if current_prefix == '':
                    new_prefix = name
                else:
                    new_prefix = current_prefix + "." + name
                module_path_name = f"{current_prefix}.{name},{type(sub_module).__name__}"
                print(module_path_name)
                assert module_path_name not in self.all_modules, f'''duplicate name for moudle list: {module_path_name}'''
                self.all_modules.update({module_path_name:sub_module})
                stack.append((sub_module, new_prefix))
                
    def get_module_device(self, module_ori_path:str):
        device_map = self.model.hf_device_map
        if ',' not in module_ori_path:
            assert module_ori_path in device_map, f'''This module:{module_ori_path} do not have a name and do not have a direct path in the device map.'''
        module_path, module_type = module_ori_path.split(',')
        module_root_path = module_path.split('.')
        if '.'.join(module_root_path) in device_map:
            device_index = str(device_map['.'.join(module_root_path)])
            print(f'''device for {module_root_path} is {'cuda:'+device_index}''')
            return 'cuda:'+device_index
        else:
            print('.'.join(module_root_path),'not in device map')
        for i in range(1,len(module_path_list)+1):
            remaining_list = module_path_list[:-i]
            if '.'.join(remaining_list) in device_map:
                device_index = str(device_map['.'.join(module_root_path)])
                print(f'''device for {module_root_path} is {'cuda:'+device_index}''')
                return 'cuda:'+device_index
            else:
                print('.'.join(remaining_list),'not in device map')
        assert False, f'''Device for module:{module_ori_path} not found please check!'''
        

    def add_single_hook(self, module_path_name:str, hook_type:str):
        assert hook_type in self.supported_hook_dict, f'''{hook_type} is not a defined hook.'''
        instance_hook = self.supported_hook_dict[hook_type](module_path_name = module_path_name)
        align_hook = hooks.AlignDevicesHook(execution_device = self.get_module_device(module_path_name), io_same_device = True)
        self.hook_module_list.append((module_path_name, self.all_modules[module_path_name], instance_hook))
        hooks.add_hook_to_module(self.all_modules[module_path_name],align_hook)
        hooks.add_hook_to_module(self.all_modules[module_path_name],instance_hook, append = True)
        print(f'''hook of {module_path_name} added.''')
        
    def add_hook_for_each_layer_by_name(self, 
                                        hook_type:str,
                                        module_name:str,
                                        layer_list:List[int]):
        '''Add hook to repeated sepecific module in each layers like mlp......'''
        for layer_id in layer_list:
            for module_path_name in self.all_modules.keys():
                if module_path_name.startswith(f'model.layers.{layer_id},{module_name}'):
                    self.add_single_hook(module_path_name, hook_type)
                    
    def add_hook_to_layer(self, 
                          hook_type:str,
                          layer_list:List[int],):
        for layer_id in layer_list:
            for module_path_name in self.all_modules.keys():
                if module_path_name.startswith(f'model.layers.{layer_id},'):
                    self.add_single_hook(module_path_name, hook_type)
        
                    
    def remove_all_hooks(self):
        for hooked_module in self.hook_module_list:
            hooks.remove_hook_from_module(hooked_module[1])
        self.hook_module_list = []
        
    def set_hook_parameters(self, set_layer:Optional[List[int]] = None, **kwargs):
        keyword_args = ['mode', 'control_intensity', 'direction_name', 'direction_name']
        for hook_module_index, hooked_module in enumerate(self.hook_module_list):
            if set_layer is None:
                pass
            else:
                assert isinstance(set_layer, list), f'''Set layer should be a list'''
                if hook_module_index in set_layer:
                    pass # set the parameter as specified
                else: # do not set paramter for these layers and freeze hooks for these not specified layer
                    setattr(hooked_module[2], 'mode', 'freeze')
                    continue
                    
            for parameter in keyword_args:
                new_parameter = kwargs.get(parameter)
                if new_parameter is None: # don't change this paramter
                    pass
                else:
                    setattr(hooked_module[2], parameter, new_parameter)
                    if parameter == 'mode':
                        self.mode = new_parameter

    def calculate_hook_direction(self, direction_name_pos:str, direction_neg_name:Optional[str] = None):
        for hooked_module in tqdm(self.hook_module_list, desc="Calculating direction components"):
            assert len(self.hiddenstate_label)>0, f'''currently no data in self.hiddenstate_label, should pass input_data_label when forward with train mode'''
            if direction_neg_name is None:
                tuple_direction_name = (direction_name_pos, ) # for concept
            else:
                tuple_direction_name = (direction_name_pos, direction_neg_name) # for function
            hooked_module[2].calculate_direction(self.hiddenstate_label, tuple_direction_name)

    def clear_hook_cache(self):
        for hooked_module in self.hook_module_list:
            hooked_module[2].pre_hook_cache = []
            hooked_module[2].post_hook_cache = []
        self.hiddenstate_label = []
        
    def get_model_emotion_desc(self, not_want_emotion: List[str] = []):
        direc_description = {
            ('batman', 'joker'): 'This is a set of opposing role-playing tones. If you feel that role-playing can enhance the user experience during interaction, choose this tone. If you think a somewhat naive sense of justice is more appropriate, select Batman. If you prefer a slightly evil tone, then choose the Joker.',
            ('honest', 'dishonest'): 'This is a pair of honest and dishonest tones. If you think it is better to conceal the truth in your current reply—for example, not telling a patient they have cancer—then choose the dishonest tone. If you believe the user wants to be treated honestly, please choose the honest tone.',
            ('happiness',): 'If you feel that responding with happiness would enhance the user experience, please select this tone.',
            ('sadness',): 'If you believe that a sad tone would enhance the interaction, please choose this tone.',
            ('anger',): 'If you think that an angry tone would make the conversation more engaging, please choose this tone.',
            ('fear',): 'If you believe that using a tone that induces fear or terror would enhance the user’s experience, please select this tone.',
            ('disgust',): 'If you feel that expressing disgust would enhance the user experience, please choose this tone.',
            ('surprise',): 'If you think that expressing surprise would make the conversation more interesting, please choose this tone.',
                        }
        all_directions = self.hook_module_list[0][2].control_directions.keys()
        direction_desc_string = ''
        for key in all_directions:
            for item in not_want_emotion:
                if item in key:
                    continue
            if key not in direc_description:
                assert False, f'''{key} is not in your description dictionary, please add description for this empotion'''
            direction_desc_string += f'{key}: {direc_description[key]}\n'
        return direction_desc_string
            
        
    def train_with_data(self, batch_size, direction_name, X, Y):
        
        for i in tqdm(range(0, len(X), batch_size), desc=f"Inferencing: {direction_name}"):
            encodings = self.tokenizer(X[i:i+batch_size], return_tensors='pt', padding=True, truncation=True)
            input_ids = encodings['input_ids'].to(self.model.device)
            attention_mask = encodings['attention_mask'].to(self.model.device)
            with torch.no_grad():
                outputs = self(input_ids=input_ids, attention_mask=attention_mask, input_data_label = Y[i:i+batch_size], use_cache = False)
        assert isinstance(direction_name, tuple), f'''input direction name should be a tuple (+, -)'''
        if len(direction_name) == 2:
            self.calculate_hook_direction(direction_name[0], direction_name[1])
        elif len(direction_name) == 1:
            self.calculate_hook_direction(direction_name[0])
        else:
            assert False, f'''Length of input direction should be 1 or 2 but your direction {direction_name} have len: {len(direction_name)}'''
        self.clear_hook_cache()
        torch.cuda.empty_cache()
        
