from convlab2.dpl.etc.util.vector_diachat import DiachatVector
from convlab2.dpl.etc.loader.unused.loader_mle import ActMLEPolicyDataLoader


class ActMLEPolicyDataLoaderDiachat(ActMLEPolicyDataLoader):

    def __init__(self):
        self.vector = DiachatVector()

        processed_dir = 'convlab2/dpl/etc/data'
        # if os.path.exists(processed_dir):
        #     print('Load processed data file')
        #     self._load_data(processed_dir)
        # else:
        print('Start preprocessing the dataset')
        self._build_data(processed_dir)
