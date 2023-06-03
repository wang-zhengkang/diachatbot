def delexicalize_da(da):
    delexicalized_da = []  #act-domain-slot
    delexicalized_act = []
    delexicalized_domain = []
    delexicalized_act_domain = []
    
    
    for intent, domain, slot, _ in da:
        deda = []
        de_act_domain = []
        
        deda.append(intent)
        deda.append('none' if domain == '' else domain)
        deda.append('none' if slot == '' else slot)
        delexicalized_da.append('-'.join(deda))
        
        delexicalized_act.append(intent)
        delexicalized_domain.append('none' if domain == '' else domain)
        
        de_act_domain.append(intent)
        de_act_domain.append('none' if domain == '' else domain)
        delexicalized_act_domain.append('-'.join(de_act_domain))
        
    delexicalized_da = list(set(delexicalized_da))
    delexicalized_act = list(set(delexicalized_act))
    delexicalized_domain = list(set(delexicalized_domain))
    delexicalized_act_domain = list(set(delexicalized_act_domain))
        
    return delexicalized_act,delexicalized_domain,delexicalized_act_domain,delexicalized_da