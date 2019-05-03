import argparse
import configparser
import random
import numpy as np

# Convenient lookup table for various paths.
# NOTE: Also used by the parser in main.
data_paths = {
    # "trial_answers": 'friends/Answers_for_trial_data/answer.txt',    # No longer used
    "trial": {"episode": 'data/friends/Trial_and_Training_data_and_Entity_mapping/friends.trial.episode_delim.conll',
              "scene": 'data/friends/Trial_and_Training_data_and_Entity_mapping/friends.trial.scene_delim.conll',
              },
    "train": {"episode": 'data/friends/Trial_and_Training_data_and_Entity_mapping/friends.train.episode_delim.conll',
              "scene": 'data/friends/Trial_and_Training_data_and_Entity_mapping/friends.train.scene_delim.conll',
              },
    "test": {"episode": 'data/friends/testing_data/friends.test.episode_delim.conll',
             "scene": 'data/friends/testing_data/friends.test.scene_delim.conll',
             },
    "entity_map": 'data/friends/Trial_and_Training_data_and_Entity_mapping/friends_entity_map.txt',
    "embeddings": {"google_news": 'data/GoogleNews-vectors-negative300.bin.gz',
# TODO Delete the following in published version.
               # TODO @Future: set path to original file, s.t., given a list of relevant entity names, filtering (as done with token embeddings) can be applied (more general)
               # TODO @Future: I'd rather put these embeddings not in friends/ but in data/ ?
               "freebase": 'data/friends/Trial_and_Training_data_and_Entity_mapping/main_freebase.txt',
               "google_token_embs_scene": 'data/friends/Trial_and_Training_data_and_Entity_mapping/friends.train.scene_delim__GoogleNews-vectors-negative300.npy',
               },
    }


def settings_from_config(file, random_sample=False):
    config = read_config_file(file)

    sampled_params = {}
    fixed_params = {}

    settings = {}
    for section in config.keys():
        settings[section.lower()] = {}
        for param in config[section].keys():
            value = None
            # Either use defaults or sample randomly in the right manner
            if not random_sample or 'sample' not in config[section][param]:
                value = config[section][param]['default']
                fixed_params[param] = value
            else:
                values = config[section][param]['sample']
                if isinstance(values, list):
                    value = random.choice(values)
                elif isinstance(values, tuple):
                    if config[section][param]['type'] == int:
                        value = random.randint(values[0], values[1])
                    else:
                        if len(values) > 2 and values[2] == 'log':
                            minimum = max(values[0], 1e-7)  # Avoid error caused by log 0
                            value = 10 ** random.uniform(np.log10(minimum), np.log10(values[1] + 1e-10))
                        else:
                            value = random.uniform(values[0], values[1])
                    ## Copied from old version, in case reimplementing quantized sampling is desired.
                    # gen_randvalue = lambda interval: \
                    #     random.randint(interval[0], interval[1]) if isinstance(interval[0], int) \
                    #         else decimal.Decimal(random.uniform(interval[0], interval[1] + 1e-10)).quantize(
                    #         decimal.Decimal("%s" % interval[0]))
                sampled_params[param] = value
            settings[section.lower()][param] = value
        # Turn dict into namespace
        settings[section.lower()] = argparse.Namespace(**settings[section.lower()])
    # Turn dict into namespace
    settings = argparse.Namespace(**settings)

    # Shortcuts for data directories:
    if str(settings.data.dataset) in data_paths:
        settings.data.dataset = data_paths[settings.data.dataset][settings.data.level]
    if not 'vocabulary' in vars(settings.data):
        settings.data.vocabulary = settings.data.dataset.replace('.conll', '.vocab')
    if not 'entity_map' in vars(settings.data):
        settings.data.entity_map = data_paths['entity_map']
    if not 'folds_dir' in vars(settings.data):
        settings.data.folds_dir = settings.data.dataset.replace('.conll', '_{0}_fold.pkl'.format(settings.data.folds))
    if str(settings.model.token_emb) in data_paths['embeddings']:
        settings.model.token_emb = data_paths['embeddings'][settings.model.token_emb]
    if str(settings.model.speaker_emb) in data_paths['embeddings']:
        settings.model.speaker_emb = data_paths['embeddings'][settings.model.speaker_emb]

    # TODO Remove the following at some point; for backward compatibility only:
    if not 'entlib_value_weights' in vars(settings.model):
        settings.model.entlib_value_weights = settings.model.entlib_weights

    # NOTE: The following is not fully general, and perhaps it shouldn't be (for data paths etc...)
    # Do this here (before checking for nonsensical combinations) in order for directory name to remain constant.
    for key in vars(settings.model):
        if key in fixed_params:
            fixed_params[key] = vars(settings.model)[key]

    # TODO @Future: Don't forget to update the following as more hyperparameters are added.
    # Correct nonsensical (combinations of) settings:
    if settings.model.bidirectional == True:
        settings.model.hidden_lstm_1 += settings.model.hidden_lstm_1 % 2    # Make even.
    if settings.model.entity_library == 'static':
        settings.model.entlib_weights = True        # static without weights makes no sense (always zero zero zero ...)
        settings.model.entlib_value_weights = True  # and in the static case, the values just ARE the weights.
    if settings.model.entlib_shared == True:
        settings.model.entlib_sharedinit = True

    # Set unused settings to None; for cleaner file names and for convenience in analysis.py.
    if settings.model.attention_lstm == False:
        settings.model.attention_window = None
        settings.model.window_size = None
        settings.model.nonlinearity_a = None
    if settings.model.attention_window == False:
        settings.model.window_size = None
        settings.model.nonlinearity_a = None
    if settings.model.entity_library == False:
        settings.model.entlib_weights = None
        settings.model.entlib_shared = None
        settings.model.entlib_key = None
        settings.model.gate_nonlinearity = None
        settings.model.gate_type = None
        settings.model.gate_softmax = None
        settings.model.entlib_normalization = None
        settings.model.entlib_value_weights = None
    if settings.model.entlib_weights == False:
        settings.model.entlib_shared = None
        settings.model.entlib_key = None
        settings.model.entlib_value_weights = None
    if settings.model.gate_type != 'mlp':
        settings.model.gate_mlp_hidden = None
    if settings.model.entity_library == 'static':
        settings.model.entlib_normalization = None
        settings.model.entlib_key = None
    if settings.model.entlib_key is None or settings.model.entlib_key == False:
        settings.model.gate_sum_keys_values = None
    if settings.model.entlib_shared is None:
        settings.model.entlib_sharedinit = None

    # NOTE: The following is not fully general, and perhaps it shouldn't be (for data paths etc...)
    for key in vars(settings.model):
        if key in sampled_params:
            sampled_params[key] = vars(settings.model)[key]

    return settings, fixed_params, sampled_params


def read_config_file(file):
    config = configparser.ConfigParser()
    config.read(file)
    settings = {}
    for section in config.sections():
        settings[section] = {}
        for option in config.options(section):
            value = config.get(section,option)
            partition = value.partition('#')
            main = partition[0].split()
            option = option.replace(" ","_")
            default, type = _value_reader(main[0], typed=True)
            settings[section][option] = {'default': default, 'type': type}
            settings[section][option].update([_value_reader(s) for s in main[1:]])
            settings[section][option]['help'] = partition[2].strip()
    return settings


def write_config_file(args, file_name):
    config = configparser.ConfigParser()
    for section in vars(args).keys():
        config.add_section(section)
        for option, value in vars(vars(args)[section]).items():
            if isinstance(value, bool) and value == True: value = 'yes'
            elif isinstance(value, bool) and value == False: value = 'no'
            config.set(section, option.replace('_', ' '), str(value))
    with open(file_name, 'w') as configfile:
        config.write(configfile)


def _value_reader(s, typed=False):
    """
    Converts string to bool if possible, otherwise int, otherwise float, otherwise list/interval, otherwise string
    :param s: a string
    :return: the string's interpretation
    """
    if s.lower() == 'none':
        s = None
        return (s, None) if typed else s
    if s == 'yes' or s == 'no':
        s = (s == 'yes')
        return (s, 'bool') if typed else s
    if s.count('.') == 0:       # Is it an int?
        try:
            return (int(s), int) if typed else int(s)
        except ValueError:  # Apparently not...
            pass
    try:            # Is it a float, perhaps?
        return (float(s), float) if typed else float(s)
    except ValueError:
        pass        # Nope. That means it's a string:
    # TODO @Future: the following would be safer with regular expression matching
    if '-' in s and len(s.split('-'))==2:    # A linear interval?
        s = [_value_reader(v, False) for v in s.split('-')]
        return ('sample', (s[0], s[1], 'lin'))
    elif '~' in s and len(s.split('~'))==2:  # A log interval?
        s = [_value_reader(v, False) for v in s.split('~')]
        return ('sample', (s[0], s[1], 'log'))
    elif '|' in s:  # A set of options?
        s = [_value_reader(v, False) for v in s.split('|')]
        return ('sample', s)

    return (s, str) if typed else s       # Ordinary string then


def fixed_params_to_string(file):
    _, fixed_params, _ = settings_from_config(file, random_sample=True)
    stringed_params = []
    for key in sorted(fixed_params):
        if not (key == 'phase' or key == 'stop_criterion' or key == 'test_every' or key == 'layers_lstm'):
            if isinstance(fixed_params[key],bool):
                if fixed_params[key]:
                    stringed_params.append(key[:3])
            elif key == 'level' or key == 'optimizer':
                stringed_params.append(fixed_params[key][:3])
            else:
                stringed_params.append(key[:3] + str(fixed_params[key])[:3])
    return '-'.join(stringed_params)

# TODO @Future merge this function with the foregoing. Also, file naming could be improved wrt readability.
def params_to_string(params):
    stringed_params = []
    for key in sorted(params):
        if not (key == 'phase' or key == 'stop_criterion' or key == 'test_every' or key == 'layers_lstm'):
            if params[key] is not None:
                if isinstance(params[key],bool):
                    stringed_params.append(key[:3] + str(params[key])[0])
                elif key == 'level' or key == 'optimizer':
                    stringed_params.append(params[key][:3])
                else:
                    stringed_params.append(key[:3] + str(params[key])[:3])
    return '-'.join(stringed_params)
