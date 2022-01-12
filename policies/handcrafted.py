import numpy as np
import random


def handcrafted_policy1(input_state, sort_categories_by_size=False,
                        block_base=False):
    """

    :param state:
    :return: Object ID (python ID)
    """
    # print("ln13")
    # print(state[0]['current_quats'].shape)
    # print(state[1]['current_quats'].shape)
    # print(state[2]['current_quats'].shape)

    if isinstance(input_state, np.ndarray):
        list_of_dicts = input_state.tolist()

    def get_python_id(state, sort_categories_by_size):
        print(f"ln14 {state['future_moved_from_target']}")
        if state['future_moved_from_target']:
            return np.array(-1)
        ec_list = state['extra_config']

        for dic in ec_list:
            assert dic['block_type'] in ['cap', 'support']

        # print("ln19 visited")
        # print(state['visited'])
        assert state['visited'].dtype != np.dtype('O'), state['visited'].dtype

        # get all visited python indices
        visited_python_indices = np.where(state['visited'] == 1)[0]

        # zipped has [python_indices, extra_config_dicts, volumes]
        zipped = zip(range(len(ec_list)), ec_list, state['volumes'])

        # get all extra config dicts for non-visited
        # ec_list_nonvisited = [ec_list[idx] if idx not in list_of_ints else None for idx in range(len(ec_list))]

        # get all block types for non-visited by pulling out of the extra config dicts
        # block_types_nonvisited = [ec['block_type'] if isinstance(ec, dict) else None for ec in ec_list_nonvisited]

        nonvisited_zips = [zip for zip in zipped if zip[0] not in visited_python_indices]

        nonvisited_support_zips = [zip for zip in nonvisited_zips if zip[1]['block_type'] == "support"]

        nonvisited_cap_zips = [zip for zip in nonvisited_zips if zip[1]['block_type'] == "cap"]

        def check_block_based_unvisited(zipped_list):
            for zip in zipped_list:
                obj_idx = zip[0]
                if "foundation" in state['object_names'][obj_idx]:
                    return True
            return False

        def get_unvisited_block_base(zipped_list):
            for zip in zipped_list:
                obj_idx = zip[0]
                if "foundation" in state['object_names'][obj_idx]:
                    return obj_idx
            raise NotImplementedError

        if nonvisited_support_zips and not state['capped']:
            # support zips in ascending order
            if sort_categories_by_size:
                ascending_sort = sorted(nonvisited_support_zips, key=lambda tup: tup[2])

                # get python oid of largest support zip
                obj_idx = ascending_sort[-1][0]
            elif block_base and check_block_based_unvisited(nonvisited_support_zips):
                obj_idx = get_unvisited_block_base(nonvisited_support_zips)
            else:
                # just get a random support
                shuffled_zips = random.sample(nonvisited_support_zips, k=len(nonvisited_support_zips))
                obj_idx = shuffled_zips[-1][0]
        elif nonvisited_cap_zips and not state['capped']:
            if sort_categories_by_size:
                ascending_sort = sorted(nonvisited_cap_zips, key=lambda tup: tup[2])
                obj_idx = ascending_sort[-1][0]
            else:
                # just get a random cap
                shuffled_zips = random.sample(nonvisited_cap_zips, k=len(nonvisited_cap_zips))
                obj_idx = shuffled_zips[-1][0]
        else:
            obj_idx = -1

        if obj_idx >= 0:
            assert state['visited'][obj_idx] != 1, (obj_idx, state['visited'])

        assert isinstance(obj_idx, int), obj_idx

        print(f"ln74 gt_a object name: {state['object_names'][obj_idx] if obj_idx >= 0 else 'Noop'}")
        return np.array(obj_idx)

    if isinstance(input_state, np.ndarray):
        return np.array([dict(selected_object_id=get_python_id(state_, sort_categories_by_size=sort_categories_by_size))
                         for state_ in list_of_dicts])

    if get_python_id(input_state, sort_categories_by_size) == -1 and input_state['timestep'] == 0:
        import pdb
        pdb.set_trace()
    return dict(selected_object_id=get_python_id(input_state, sort_categories_by_size))