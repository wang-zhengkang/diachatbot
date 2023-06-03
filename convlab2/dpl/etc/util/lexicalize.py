from copy import deepcopy


def delexicalize_da(da):
    delexicalized_da = []
    delexicalized_act = []
    delexicalized_domain = []
    delexicalized_ad = []

    for act, domain, slot, _ in da:
        deda = []
        de_act_domain = []

        deda.append(act)
        deda.append('none' if domain == '' else domain)
        deda.append('none' if slot == '' else slot)
        delexicalized_da.append('-'.join(deda))

        delexicalized_act.append(act)
        delexicalized_domain.append('none' if domain == '' else domain)

        de_act_domain.append(act)
        de_act_domain.append('none' if domain == '' else domain)
        delexicalized_ad.append('-'.join(de_act_domain))

    delexicalized_da = list(set(delexicalized_da))
    delexicalized_act = list(set(delexicalized_act))
    delexicalized_domain = list(set(delexicalized_domain))
    delexicalized_ad = list(set(delexicalized_ad))

    return delexicalized_act, delexicalized_domain, delexicalized_ad, delexicalized_da


def deflat_da(meta):
    meta = deepcopy(meta)
    dialog_act = []
    for da in meta:
        act, domain, slot = da.split('-')
        adsv = [act, domain, slot, 'none']
        dialog_act.append(adsv)
    return dialog_act
