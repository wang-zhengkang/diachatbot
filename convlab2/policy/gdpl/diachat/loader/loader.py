import os
from convlab2.policy.gdpl.diachat.util.vector_diachat import DiachatVector
from convlab2.policy.gdpl.diachat.loader.loader_mle import ActMLEPolicyDataLoader


class ActMLEPolicyDataLoaderDiachat(ActMLEPolicyDataLoader):

    def __init__(self):
        sys_da_file = 'convlab2/policy/gdpl/diachat/data/sys_da.json'
        usr_da_file = 'convlab2/policy/gdpl/diachat/data/usr_da.json'
        self.vector = DiachatVector(sys_da_file, usr_da_file)

        processed_dir = 'convlab2/policy/gdpl/diachat/data'
        # if os.path.exists(processed_dir):
        #     print('Load processed data file')
        #     self._load_data(processed_dir)
        # else:
        print('Start preprocessing the dataset')
        self._build_data(processed_dir)
