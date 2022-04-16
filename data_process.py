import os
os.environ['TF_KERAS']='1'
import json
path = r'F:\PyProject\BERT模型\chinese_roformer-char_L-12_H-768_A-12'
config_path = path + '\\bert_config.json'
checkpoint_path = path + '\\bert_model.ckpt'
dict_path = path + '\\vocab.txt'

from bert4keras.tokenizers import Tokenizer

# create text tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

#key roles
event_key_role_dict = {'EquityPledge':'PledgedShares',
                      'EquityRepurchase':'RepurchasedShares',
                      'EquityOverweight':'TradedShares',
                      'EquityUnderweight':'TradedShares',
                      'EquityFreeze':'FrozeShares'}
#sub-key roles
event_key_role_dict2 = {'EquityPledge':'Pledgee',
                      'EquityRepurchase':'LowestTradingPrice',
                      'EquityOverweight':'EndDate',
                      'EquityUnderweight':'EndDate',
                      'EquityFreeze':'LegalInstitution'}

#all relations
relations = ['EquityOverweight-EquityHolder','EquityOverweight-TradedShares', 'EquityOverweight-StartDate', 'EquityOverweight-EndDate', 'EquityOverweight-LaterHoldingShares', 'EquityOverweight-AveragePrice',
             'EquityUnderweight-EquityHolder','EquityUnderweight-TradedShares', 'EquityUnderweight-StartDate', 'EquityUnderweight-EndDate', 'EquityUnderweight-LaterHoldingShares', 'EquityUnderweight-AveragePrice',
             'EquityFreeze-EquityHolder', 'EquityFreeze-FrozeShares','EquityFreeze-LegalInstitution', 'EquityFreeze-TotalHoldingShares', 'EquityFreeze-TotalHoldingRatio', 'EquityFreeze-StartDate', 'EquityFreeze-EndDate', 'EquityFreeze-UnfrozeDate',
             'EquityPledge-Pledger', 'EquityPledge-PledgedShares','EquityPledge-Pledgee', 'EquityPledge-TotalHoldingShares', 'EquityPledge-TotalHoldingRatio', 'EquityPledge-TotalPledgedShares', 'EquityPledge-StartDate', 'EquityPledge-EndDate', 'EquityPledge-ReleasedDate',
             'EquityRepurchase-CompanyName', 'EquityRepurchase-HighestTradingPrice', 'EquityRepurchase-LowestTradingPrice', 'EquityRepurchase-RepurchasedShares','EquityRepurchase-ClosingDate', 'EquityRepurchase-RepurchaseAmount']
#remove some unused relations
for event_key in event_key_role_dict:
    relations.remove(event_key+'-'+event_key_role_dict[event_key])
#add co-occurrence relation
relations.append('co-occurrence')

#relation to id
relations2id = {}
#id to relation
id2relation = {}
for i,r in enumerate(relations):
    relations2id[r]=i
    id2relation[i]=r

# predicate2id, id2predicate = relations2id, id2relation

def load_data(file):
    result = []
    fp = open(file,encoding='utf-8')
    datas = json.load(fp)
    event_count = 0
    for data in datas:
        data = data[1]
        sentences = data['sentences']
        text = ' '.join(sentences)
        events = data['recguid_eventname_eventdict_list']
        spo_list =[]
        for event in events:
            event_type = event[1]
            arguments = event[2]
            # print(event_type, arguments)
            key_role = event_key_role_dict[event_type]   #关键角色，对应关系主体（subject）
            subject = arguments[key_role]
            for role in arguments:
                if role != key_role:
                    relation = event_type + '-' + role
                    if subject and arguments[role]:
                        spo_list.append((subject, relation, arguments[role]))
            #添加“共事”角色
            key_role2 = event_key_role_dict2[event_type]    #第二关键角色
            role_value = arguments[key_role2]               #第二关键角色值
            items = list(arguments.items())
            for i in range(len(items)):#用于添加共事关系
                co_role_value =items[i][1]  #共事角色
                if role_value != co_role_value and role_value!=subject and co_role_value != subject:
                    if role_value and co_role_value and role_value != co_role_value:
                            if (role_value, 'co-occurrence', co_role_value) not in spo_list:
                                spo_list.append((role_value, 'co-occurrence', co_role_value))
        event_count += len(events)
        result.append({'text':text,'spo_list':spo_list})
    print(event_count)
    fp.close()
    return result
