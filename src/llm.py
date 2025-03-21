import yaml
from pathlib import Path
from nltk.corpus import wordnet as wn

from jinja2 import Template
from guidance import models, guidance, gen, assistant, system, select, substring
# from guidance.chat import Llama3dot2ChatTemplate

from .homophones import get_homophones, _download_resources

def _load_config(key: str) -> dict:
    '''Loads the configuration file.'''
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)[key]
    
    

def _template(text: str, **context: dict) -> str:
    '''Renders a template string.'''
    return Template(text).render(**context)
    
    

def init_llm(default_system: bool = True) -> models.LlamaCpp:
    '''Initializes the LLM model.'''
    _download_resources()
    llm_config = _load_config('llm')
    llama = models.LlamaCpp(
        # chat_template=Llama3dot2ChatTemplate(),
        **llm_config        
    )
    
    if default_system:
        with system():
            llama += _load_config('prompts')['default']
    
    return llama
    
@guidance
def explain_pun(lm):
    '''Explains a pun, if the user has provided one.'''
    prompts = _load_config('prompts')
    with system():
        llm = lm + prompts['explain_pun']
    with assistant():
        llm += gen('explain_pun')
        if 'explain_pun' in llm:
            return lm + llm['explain_pun']
        else:
            return lm + gen()
    
@guidance
def handle_homograph(lm, pun):
    '''Chooes the relevant definition of a homograph.'''
    prompts = _load_config('prompts')['handle_homograph']
    synset = wn.synsets(pun)
    definitions = [s.definition() for s in synset]

    if len(definitions) < 2:
        return lm + explain_pun()

    elif len(definitions) == 2:
        def1, def2 = definitions
    
    else:
        with system():
            llm = lm + _template(prompts['system'], pun=pun) + '\n - '.join(definitions)
        
        with assistant():
            llm += _template(prompts['def1'], pun=pun) + select(definitions, name='def1') + '"'
            if 'def1' in llm:
                def1 = llm['def1']
                definitions.remove(def1)
                llm += _template(prompts['def2'], pun=pun) + select(definitions, name='def2')
                if 'def2' in llm:
                    def2 = llm['def2']
                else:
                    return lm + explain_pun()
            else:
                return lm + explain_pun()
            
    with assistant():
        return lm + _template(prompts['output'], pun=pun, def1=def1, def2=def2) + gen(stop='.') + '.'
        
    
@guidance
def handle_homophone(lm, pun):
    '''Chooses the relevant homophone of a pun.'''
    prompts = _load_config('prompts')['handle_homophone']
    cleansed_pun = pun.replace('-', ' ').lower()
    cleansed_pun = ''.join([c for c in cleansed_pun if c.isalpha() or c.isspace()])
    
    similar_phones = get_homophones(cleansed_pun)
    
    if len(similar_phones) == 0:
        return lm + explain_pun()
    
    if len(similar_phones) == 1:
        homophone = similar_phones[0]
    
    else:
        with system():
            llm = lm + _template(prompts['system'], pun=pun) + '\n - '.join(similar_phones)
        
        with assistant():
            llm += _template(prompts['homophone'], pun=pun) + select(similar_phones, name='homophone')
            if 'homophone' in llm:
                homophone = llm['homophone']
            else:
                return lm + explain_pun()
        
    with assistant():
        return lm + _template(prompts['output'], pun=pun, homophone=homophone) + gen(stop='.') + '.'
    
@guidance
def banter(lm, user_input):
    '''Banterbot response to user input.'''
    prompts = _load_config('prompts')['pipeline']
    with system():
        llm = lm + prompts['should_explain_joke']
        
    with assistant():
        llm += select(['Yes', 'No'], name='contains_pun')
    
    if 'contains_pun' in llm and llm['contains_pun'] == 'Yes':
        
        with system():
            llm += prompts['extract_pun']
            
        with assistant():
            llm += 'The pun word(s) is: ' + substring(user_input, name='pun')
        
        if 'pun' in llm:
            pun = llm['pun'].strip()
            
            with system():
                llm += _template(prompts['pun_type'], pun=pun)
                
            with assistant():
                llm += f'A: The pun "{pun}" ' + select(['sounds like', 'has two'], name='pun_type')
            
            if 'pun_type' in llm:
                if llm['pun_type'] == 'sounds like':
                    return lm + handle_homophone(pun)
                else:
                    return lm + handle_homograph(pun)
        
        return lm + explain_pun()
    
    with assistant():
        return lm + gen()