from qdata.datasource import get_table_id, PipeIO, PipeInfo




class Pipeline(PipeIO):
    """
        받고자 하는 데이터를 abstract 수준으로 받아서 pipeline별로 add을 통해 추가
        run을 통해 실제로 DB연결을 통해 데이터를 한꺼번에 받아오기

        pipeline은 dictionary로 관리되며 key는 pipeline의 이름, value는 DataInfo타입의 class object로 저장   

        pipeline class object 자체는 하나만 생성하되, 목적별로 다른 데이터를 pipeline name으로 구별
        run도 name별로 그떄그때 가져오는 방식
    """
    def __init__(self, name=None):
        super().__init__()
        self.added_item = dict()
        if name is not None:
            self.load(name=name)

    def add(self, name, **kwargs):
        self.pipeline[name] = PipeInfo(**kwargs)
        self.pipeline[name].add_attr(table_id=get_table_id(name))

    def add_item(self, name, added_dict):
        self.added_item.update(added_dict)
        self.pipeline[name].item.update(added_dict)

    def update(self, name, sch_obj=None, chunksize=1000):
        pipe_dict = self.pipeline[name]
        if not hasattr(self.pipeline[name], 'schedule'):
            pipe_dict.add_attr(schedule=sch_obj)

        pipe_dict.item = self.add_item
        self._insert_to_db(pipe_dict, sch_obj, chunksize)
        self.add_item = dict()

    def run(self, name, schedule=None, chunksize=1000, mode='load_or_run'):
        """
        :param name: name of pipeline
        :param schedule: schedule object if needed
        :param chunksize: chunk size to store data
        :param mode:    'load': if fail to load, return False
                         'load_or_run': if fail to load, run store
                         ['run','store','overwrite']  : loaded or not, store (overwrite)
                         'update': run added parts (i.e. added item)
        :return:
        """
        if mode in ['load', 'load_or_run']:
            is_loaded = self.load(name)
            if is_loaded:
                return True

        if mode in ['load_or_run', 'run', 'store', 'overwrite']:
            if schedule is None:
                print('Schedule should be set.')
                return False
            self.store(name=name, sch_obj=schedule, chunksize=chunksize)
            return True

        if mode in ['update']:
            self.update(name=name, sch_obj=schedule)


        return False



