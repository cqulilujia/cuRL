# -*- coding: utf-8 -*-

def init():
    global global_dict
    global_dict = {}


def set_value(name, value):
    global_dict[name] = value


def set_value_batch(name_list, value_list):
    if str(type(name_list)) != "<class 'list'>" or str(type(value_list)) != "<class 'list'>":
        raise TypeError('name and value should be list!')
    if len(name_list) != len(value_list):
        raise Exception("name and value size should be equal!")
    for i in range(len(name_list)):
        global_dict[name_list[i]] = value_list[i]


def get_value(name, defValue=None):
    try:
        return global_dict[name]
    except KeyError:
        return defValue


def get_value_batch(name_list, defValue=None):
    if str(type(name_list)) != "<class 'list'>":
        raise TypeError('name should be list!')
    try:
        value_list = []
        for i in range(len(name_list)):
            value_list.append(global_dict[name_list[i]])
        return value_list
    except KeyError:
        return defValue