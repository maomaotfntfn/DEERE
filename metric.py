from data_process import *

#event to roles
event_type_roles_list = [('EquityFreeze',
                          ['EquityHolder', 'FrozeShares', 'LegalInstitution', 'TotalHoldingShares', 'TotalHoldingRatio',
                           'StartDate', 'EndDate', 'UnfrozeDate']),
                         ('EquityRepurchase',
                          ['CompanyName', 'HighestTradingPrice',
                          'LowestTradingPrice', 'RepurchasedShares', 'ClosingDate', 'RepurchaseAmount']),
                         (
                         'EquityUnderweight',
                         ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate', 'LaterHoldingShares',
                          'AveragePrice']),
                         ('EquityOverweight',
                                             ['EquityHolder', 'TradedShares', 'StartDate', 'EndDate',
                                              'LaterHoldingShares', 'AveragePrice']),
                         ('EquityPledge',
                                        ['Pledger', 'PledgedShares', 'Pledgee', 'TotalHoldingShares','TotalHoldingRatio',
                                                'TotalPledgedShares','StartDate', 'EndDate','ReleasedDate'])]
event_role_num_list = [len(roles) for _, roles in event_type_roles_list]            #事件
event2role_list = {event_type:roles for event_type, roles in event_type_roles_list} #事件类型->论元列表
event_types = [event_type for event_type, roles in event_type_roles_list]
event_type2id = {type:i for i,type in enumerate(event_types)}
id2event_type = {i:type for i,type in enumerate(event_types)}

def load_gold_events(file):
    format_events_list = []
    fp = open(file, encoding='utf-8')
    datas = json.load(fp)
    for data in datas:
        format_events = [None]*5
        data = data[1]
        events = data['recguid_eventname_eventdict_list']
        for event in events:
            event_type = event[1]
            arguments = event[2]
            event_id = event_type2id[event_type]
            if format_events[event_id] == None:format_events[event_id] =[]
            values = [value for value in arguments.values()]
            format_events[event_id].append(values)
        format_events_list.append(format_events)
    fp.close()
    return format_events_list

def co_event(arg1,arg2,co_spoes):
    for s,p,o in co_spoes:
        if arg1==s and arg2 ==o or arg1==o and arg2==s:
            return True
    return False

def is_single_event_cluster(event_type,args,key_role2):
    key_role2_count = len([arg for arg in args if arg[0] == key_role2])
    if key_role2_count <= 1:
        return True
    arg_dict ={}
    for role,value in args:
        if role not in arg_dict:arg_dict[role] =[]
        arg_dict[role].append(value)
    mulit_arg_count = len([value for value in arg_dict.values() if len(value)>1])
    if mulit_arg_count > len(event2role_list[event_type])*0.2:
        return False
    else:
        return True
def select_role_with_highest_score(roles_with_score):
    map = {}
    for role_value,score in roles_with_score.items():
        role,value = role_value
        if role not in map or map[role][1] < score:
            map[role] = (value,score)
    return [(role,value) for role,(value,score) in map.items()]

def get_final_events(event_type,key_role_value,args,co_spoes):
    key_role = event_key_role_dict[event_type]
    key_role2 = event_key_role_dict2[event_type]
    key_roles2_args = [arg for arg in args if arg[0] == key_role2]
    E = []
    if  is_single_event_cluster(event_type,args,key_role2):
        E = [args]
    else:
        for key_role2,key_role2_value in key_roles2_args:
            e ={(key_role,key_role_value):0,(key_role2,key_role2_value):0}
            for role,value in args:
                 if co_event(key_role2_value,value,co_spoes)==True:  #第二
                    if (key_role2_value,'co-occurrence',value) in co_spoes:
                        e[(role,value)] = co_spoes[(key_role2_value,'co-occurrence',value)]
                    else:
                        e[(role,value)] = co_spoes[(value,'co-occurrence',key_role2_value)]
            E.append(e)
    E = [ select_role_with_highest_score(e) for e in E]
    return E

#使用“同事关系进行”预测
def convert_spoes_to_events_by_co_event(R, partition=True):  # 将从文本中抽取三元组列表，转化为事件结构（事件类型，论元1，论元2，论元3...）
    events = {}  # 统计每个事件的
    co_spoes = {r:R[r] for r in R if r[1]=='co-occurrence'}
    spoes = {r:R[r] for r in R if r[1]!='co-occurrence'}
    for r in spoes:
        s, p, o = r
        score = spoes[r]
        event_type, role = p.split('-')  # 事件类型
        key = (event_type, s)  # 事件类型 + 主体论元 + 位置（每个key代表1个事件）
        sub_role = event_key_role_dict[event_type]
        if key not in events:
            events[key] = {(sub_role,s):score}
        events[key][(role,o)] = score

    format_events = [None] * 5  # 每一个事件
    for event_type_and_subject, args in events.items():
        event_type, s = event_type_and_subject
        if partition:
            final_events = get_final_events(event_type,s,args,co_spoes)
        else:
            final_events = [args]
        for final_event in final_events:
            event_id = event_type2id[event_type]  # 事件类型编号
            if format_events[event_id] == None: format_events[event_id] = []
            role_list = event2role_list[event_type]
            role_dict = {role: None for role in role_list}
            for v in final_event:
                role, role_value = v
                role_dict[role] = role_value
            event = list(role_dict.values())
            format_events[event_id].append(event)
    return format_events

def agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num):
    """
    Aggregate TP,FP,FN statistics for a single event prediction of one instance.
    A pred_records should be formated as
    [(Record Index)
        ((Role Index)
            argument 1, ...
        ), ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    role_tpfpfn_stats = [[0] * 3 for _ in range(role_num)]

    if gold_records is None:
        if pred_records is not None:  # FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
        else:  # ignore TN
            pass
    else:
        if pred_records is None:  # FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1
        else:  # True Positive at the event level
            # sort predicted event records by the non-empty count
            # to remove the impact of the record order on evaluation
            pred_records = sorted(pred_records,
                                  key=lambda x: sum(1 for a in x if a is not None),
                                  reverse=True)
            gold_records = list(gold_records)

            while len(pred_records) > 0 and len(gold_records) > 0:
                pred_record = pred_records[0]
                assert len(pred_record) == role_num

                # pick the most similar gold record
                _tmp_key = lambda gr: sum([1 for pa, ga in zip(pred_record, gr) if pa == ga])
                best_gr_idx = gold_records.index(max(gold_records, key=_tmp_key))
                gold_record = gold_records[best_gr_idx]

                for role_idx, (pred_arg, gold_arg) in enumerate(zip(pred_record, gold_record)):
                    if gold_arg is None:
                        if pred_arg is not None:  # FP at the role level
                            role_tpfpfn_stats[role_idx][1] += 1
                        else:  # ignore TN
                            pass
                    else:
                        if pred_arg is None:  # FN
                            role_tpfpfn_stats[role_idx][2] += 1
                        else:
                            if pred_arg == gold_arg:  # TP
                                role_tpfpfn_stats[role_idx][0] += 1
                            else:
                                role_tpfpfn_stats[role_idx][1] += 1
                                role_tpfpfn_stats[role_idx][2] += 1

                del pred_records[0]
                del gold_records[best_gr_idx]

            # remaining FP
            for pred_record in pred_records:
                assert len(pred_record) == role_num
                for role_idx, arg_tup in enumerate(pred_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][1] += 1
            # remaining FN
            for gold_record in gold_records:
                assert len(gold_record) == role_num
                for role_idx, arg_tup in enumerate(gold_record):
                    if arg_tup is not None:
                        role_tpfpfn_stats[role_idx][2] += 1

    return role_tpfpfn_stats

def agg_ins_event_role_tpfpfn_stats(pred_record_mat, gold_record_mat, event_role_num_list):
    """
    Aggregate TP,FP,FN statistics for a single instance.
    A record_mat should be formated as
    [(Event Index)
        [(Record Index)
            ((Role Index)
                argument 1, ...
            ), ...
        ], ...
    ], where argument 1 should support the '=' operation and the empty argument is None.
    """
    assert len(pred_record_mat) == len(gold_record_mat)
    # tpfpfn_stat: TP, FP, FN
    event_role_tpfpfn_stats = []
    for event_idx, (pred_records, gold_records) in enumerate(zip(pred_record_mat, gold_record_mat)):
        role_num = event_role_num_list[event_idx]
        role_tpfpfn_stats = agg_event_role_tpfpfn_stats(pred_records, gold_records, role_num)
        event_role_tpfpfn_stats.append(role_tpfpfn_stats)

    return event_role_tpfpfn_stats

def get_prec_recall_f1(tp, fp, fn):
    a = tp + fp
    prec = tp / a if a > 0 else 0
    b = tp + fn
    rec = tp / b if b > 0 else 0
    if prec > 0 and rec > 0:
        f1 = 2.0 / (1 / prec + 1 / rec)
    else:
        f1 = 0
    return prec, rec, f1

def event_evaluate(pred_record_mat_list,gold_record_mat_list,avg_type='micro'):
    total_event_role_stats = [
        [
            [0]*3 for _ in range(role_num)
        ] for event_idx, role_num in enumerate(event_role_num_list)
    ]

    assert len(pred_record_mat_list) == len(gold_record_mat_list)
    for pred_record_mat, gold_record_mat in zip(pred_record_mat_list, gold_record_mat_list):
        event_role_tpfpfn_stats = agg_ins_event_role_tpfpfn_stats(
            pred_record_mat, gold_record_mat, event_role_num_list
        )
        for event_idx, role_num in enumerate(event_role_num_list):
            for role_idx in range(role_num):
                for sid in range(3):
                    total_event_role_stats[event_idx][role_idx][sid] += \
                        event_role_tpfpfn_stats[event_idx][role_idx][sid]

    per_role_metric = []
    per_event_metric = []

    num_events = len(event_role_num_list)
    g_tpfpfn_stat = [0] * 3
    g_prf1_stat = [0] * 3
    event_role_eval_dicts = []
    for event_idx, role_num in enumerate(event_role_num_list):
        event_tpfpfn = [0] * 3  # tp, fp, fn
        event_prf1_stat = [0] * 3
        per_role_metric.append([])
        role_eval_dicts = []
        for role_idx in range(role_num):
            role_tpfpfn_stat = total_event_role_stats[event_idx][role_idx][:3]
            role_prf1_stat = get_prec_recall_f1(*role_tpfpfn_stat)
            per_role_metric[event_idx].append(role_prf1_stat)
            for mid in range(3):
                event_tpfpfn[mid] += role_tpfpfn_stat[mid]
                event_prf1_stat[mid] += role_prf1_stat[mid]

            role_eval_dict = {
                'RoleType': event_type_roles_list[event_idx][1][role_idx],
                'Precision': role_prf1_stat[0],
                'Recall': role_prf1_stat[1],
                'F1': role_prf1_stat[2],
                'TP': role_tpfpfn_stat[0],
                'FP': role_tpfpfn_stat[1],
                'FN': role_tpfpfn_stat[2]
            }
            role_eval_dicts.append(role_eval_dict)

        for mid in range(3):
            event_prf1_stat[mid] /= role_num
            g_tpfpfn_stat[mid] += event_tpfpfn[mid]
            g_prf1_stat[mid] += event_prf1_stat[mid]

        micro_event_prf1 = get_prec_recall_f1(*event_tpfpfn)
        macro_event_prf1 = tuple(event_prf1_stat)
        if avg_type.lower() == 'micro':
            event_prf1_stat = micro_event_prf1
        elif avg_type.lower() == 'macro':
            event_prf1_stat = macro_event_prf1
        else:
            raise Exception('Unsupported average type {}'.format(avg_type))

        per_event_metric.append(event_prf1_stat)

        event_eval_dict = {
            'EventType': event_type_roles_list[event_idx][0],
            'MacroPrecision': macro_event_prf1[0],
            'MacroRecall': macro_event_prf1[1],
            'MacroF1': macro_event_prf1[2],
            'MicroPrecision': micro_event_prf1[0],
            'MicroRecall': micro_event_prf1[1],
            'MicroF1': micro_event_prf1[2],
            'TP': event_tpfpfn[0],
            'FP': event_tpfpfn[1],
            'FN': event_tpfpfn[2],
        }
        event_role_eval_dicts.append((event_eval_dict, role_eval_dicts))

    micro_g_prf1 = get_prec_recall_f1(*g_tpfpfn_stat)
    macro_g_prf1 = tuple(s / num_events for s in g_prf1_stat)
    if avg_type.lower() == 'micro':
        g_metric = micro_g_prf1
    else:
        g_metric = macro_g_prf1

    g_eval_dict = {
        'MacroPrecision': macro_g_prf1[0],
        'MacroRecall': macro_g_prf1[1],
        'MacroF1': macro_g_prf1[2],
        'MicroPrecision': micro_g_prf1[0],
        'MicroRecall': micro_g_prf1[1],
        'MicroF1': micro_g_prf1[2],
        'TP': g_tpfpfn_stat[0],
        'FP': g_tpfpfn_stat[1],
        'FN': g_tpfpfn_stat[2],
    }
    return g_eval_dict

def is_multi_event(format_event):
    event_cnt = 0
    for event_objs in format_event:
        if event_objs is not None:
            event_cnt += len(event_objs)
            if event_cnt > 1:
                return True
    return False

def get_single_and_multi_events(gold_events_list,pred_events_list):
    single_gold_events_list=[]
    single_pred_events_list =[]
    multi_gold_events_list=[]
    multi_pred_events_list =[]
    for gold_events,pred_events in zip(gold_events_list,pred_events_list):
        if is_multi_event(gold_events):
            multi_gold_events_list.append(gold_events)
            multi_pred_events_list.append(pred_events)
        else:
            single_gold_events_list.append(gold_events)
            single_pred_events_list.append(pred_events)
    return multi_gold_events_list, multi_pred_events_list, single_gold_events_list, single_pred_events_list

def evaluate_event_all(pred_events_list,gold_test_events_list):
    g_eval_dict = event_evaluate(pred_events_list, gold_test_events_list )
    print('all', g_eval_dict)
    multi_gold_events_list, multi_pred_events_list, single_gold_events_list, single_pred_events_list = get_single_and_multi_events(
        gold_test_events_list, pred_events_list)
    g_eval_dict = event_evaluate( multi_pred_events_list, multi_gold_events_list)
    print('multi', g_eval_dict)
    g_eval_dict = event_evaluate( single_pred_events_list, single_gold_events_list)
    print('single', g_eval_dict)

gold_dev_events_list = load_gold_events(r'Data\dev.json')
gold_test_events_list = load_gold_events(r'Data\test.json')
