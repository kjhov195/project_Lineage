import os
import pandas as pd
import numpy as np
import time as sec
from sklearn.model_selection import train_test_split
from collections import Counter
from datetime import *

############### input ################
path_raw = "../raw"
path_preprocess = "../preprocess"
######################################


def preproc(fname,range_width):
    ## 경로 변경(raw data 경로))
    os.chdir(path_raw)
    ## 저장할 파일 이름
    file_name = '%s_preprocess_1.csv' %fname
    ## raw data
    df_acti_origin = pd.read_csv("%s_activity.csv" %fname, engine='python', encoding='CP949')
    df_comb_origin = pd.read_csv("%s_combat.csv" %fname, engine='python', encoding='CP949')
    df_paym_origin = pd.read_csv("%s_payment.csv" %fname, engine='python', encoding='CP949')
    df_pled_origin = pd.read_csv("%s_pledge.csv" %fname, engine='python', encoding='CP949')
    df_trad_origin = pd.read_csv("%s_trade.csv" %fname, engine='python', encoding='CP949')


    ### 1. activity
    ## tran_activity.csv 파일
    df_merged_origin = df_acti_origin

    ## num_of_characters : acc_id 당 char_id의 수
    num_of_characters = df_merged_origin.groupby(['acc_id'])['char_id'].nunique().reset_index(name='num_of_characters')
    ## num_of_days : acc_id 당 접속한 날의 수
    num_of_days = df_merged_origin.groupby(['acc_id'])['day'].nunique().reset_index(name='num_of_days')
    ## num_of_servers : acc_id 당 접속한 서버의 수
    num_of_servers = df_merged_origin.groupby(['acc_id'])['server'].nunique().reset_index(name='num_of_servers')
    ## sum_of_playtime : acc_id 당 접속시간의 합
    sum_of_playtime = df_merged_origin.groupby(['acc_id'])['playtime'].sum().reset_index(name='sum_of_playtime')
    ## sum_of_npckill : acc_id 당 npc kill의 합
    sum_of_npckill = df_merged_origin.groupby(['acc_id'])['npc_kill'].sum().reset_index(name='sum_of_npckill')
    ## sum_of_soloexp : acc_id 당 solo 경험치 획득한 양의 합
    sum_of_soloexp = df_merged_origin.groupby(['acc_id'])['solo_exp'].sum().reset_index(name='sum_of_soloexp')
    ## sum_of_partyexp : acc_id 당 party 경험치 획득한 양의 합
    sum_of_partyexp = df_merged_origin.groupby(['acc_id'])['party_exp'].sum().reset_index(name='sum_of_partyexp')
    ## sum_of_questexp : acc_id 당 quest 경험치 획득한 양의 합
    sum_of_questexp = df_merged_origin.groupby(['acc_id'])['quest_exp'].sum().reset_index(name='sum_of_questexp')
    ## sum_of_boss_days : acc_id 당 rich_monster 사냥 횟수의 합
    sum_of_boss_days = df_merged_origin.groupby(['acc_id'])['rich_monster'].sum().reset_index(name='sum_of_boss_days')
    ## sum_of_death : acc_id 당 death 횟수의 합
    sum_of_death = df_merged_origin.groupby(['acc_id'])['death'].sum().reset_index(name='sum_of_death')
    ## sum_of_revive : acc_id 당 revive 횟수의 합
    sum_of_revive = df_merged_origin.groupby(['acc_id'])['revive'].sum().reset_index(name='sum_of_revive')
    ## sum_of_exp_recovery : acc_id 당 경험치 복구 횟수의 합
    sum_of_exp_recovery = df_merged_origin.groupby(['acc_id'])['exp_recovery'].sum().reset_index(name='sum_of_exp_recovery')
    ## sum_of_fishing : acc_id 당 낚시 횟수의 합
    sum_of_fishing = df_merged_origin.groupby(['acc_id'])['fishing'].sum().reset_index(name='sum_of_fishing')
    ## sum_of_private_shop : acc_id 당 개인상점 운영 시간의 합
    sum_of_private_shop = df_merged_origin.groupby(['acc_id'])['private_shop'].sum().reset_index(name='sum_of_private_shop')
    ## num_of_game_money_change : acc_id 당 아데나 변동 횟수의 합
    num_of_game_money_change = (df_merged_origin.groupby(['acc_id'])['game_money_change'].nunique()-1).reset_index(name='num_of_game_money_change')
    ## sum_of_enchant_count : acc_id 당 7 레벨 이상 아이템 인첸트 시도 횟수의 합
    sum_of_enchant_count = df_merged_origin.groupby(['acc_id'])['enchant_count'].sum().reset_index(name='sum_of_enchant_count')

    df_acti = pd.DataFrame(df_merged_origin['acc_id'].unique(), columns = ['acc_id'])
    df_acti = pd.merge(df_acti, num_of_characters, on='acc_id')
    df_acti = pd.merge(df_acti, num_of_days, on='acc_id')
    df_acti = pd.merge(df_acti, num_of_servers, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_playtime, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_npckill, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_soloexp, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_partyexp, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_questexp, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_boss_days, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_death, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_revive, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_exp_recovery, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_fishing, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_private_shop, on='acc_id')
    df_acti = pd.merge(df_acti, num_of_game_money_change, on='acc_id')
    df_acti = pd.merge(df_acti, sum_of_enchant_count, on='acc_id')

    ## day_width : 사용할 날짜 구간
    day_width = range_width[1]-range_width[0]+1

    ## 날짜 별로 변수들을 할당
    unique_id =  df_merged_origin['acc_id'].drop_duplicates().sort_values().reset_index(drop=True)
    acc_ids = pd.DataFrame(pd.concat([unique_id]*day_width)).reset_index(drop=True)
    days = pd.Series(np.repeat(np.arange(1, day_width+1), len(unique_id)))
    acc_ids['day'] = days
    id_day_base = acc_ids.sort_values(['acc_id', 'day']).reset_index(drop=True)

    ## num_of_characters_by_day : acc_id당 각 day에 접속한 char_id의 수
    num_of_characters_by_day = df_merged_origin.groupby(['acc_id','day'])['char_id'].nunique().reset_index(name='num_of_characters_by_day')
    ## num_of_days_by_day : acc_id당 각 day의 접속 여부
    num_of_days_by_day = df_merged_origin.groupby(['acc_id','day'])['day'].nunique().reset_index(name='num_of_days_by_day')
    ## num_of_servers_by_day : acc_id당 각 day에 접속한 서버의 수
    num_of_servers_by_day = df_merged_origin.groupby(['acc_id','day'])['server'].nunique().reset_index(name='num_of_servers_by_day')
    ## sum_of_playtime_by_day : acc_id당 각 day에 플레이한 playtime의 합
    sum_of_playtime_by_day = df_merged_origin.groupby(['acc_id','day'])['playtime'].sum().reset_index(name='sum_of_playtime_by_day')
    ## sum_of_npckill_by_day : acc_id당 각 day에 죽인 npc 수의 합
    sum_of_npckill_by_day = df_merged_origin.groupby(['acc_id','day'])['npc_kill'].sum().reset_index(name='sum_of_npckill_by_day')
    ## sum_of_soloexp_by_day : acc_id당 각 day에 얻은 solo 경험치의 합
    sum_of_soloexp_by_day = df_merged_origin.groupby(['acc_id','day'])['solo_exp'].sum().reset_index(name='sum_of_soloexp_by_day')
    ## sum_of_partyexp_by_day : acc_id당 각 day에 얻은 party 경험치의 합
    sum_of_partyexp_by_day = df_merged_origin.groupby(['acc_id','day'])['party_exp'].sum().reset_index(name='sum_of_partyexp_by_day')
    ## sum_of_questexp_by_day : acc_id당 각 day에 얻은 quest 경험치의 합
    sum_of_questexp_by_day = df_merged_origin.groupby(['acc_id','day'])['quest_exp'].sum().reset_index(name='sum_of_questexp_by_day')
    ## sum_of_boss_days_by_day : acc_id당 각 day에 rich monster 사냥 횟수의 합
    sum_of_boss_days_by_day = df_merged_origin.groupby(['acc_id','day'])['rich_monster'].sum().reset_index(name='sum_of_boss_days_by_day')
    ## sum_of_death_by_day : acc_id당 각 day에 죽은 횟수의 합
    sum_of_death_by_day = df_merged_origin.groupby(['acc_id','day'])['death'].sum().reset_index(name='sum_of_death_by_day')
    ## sum_of_revive_by_day : acc_id당 각 day에 부활한 횟수의 합
    sum_of_revive_by_day = df_merged_origin.groupby(['acc_id','day'])['revive'].sum().reset_index(name='sum_of_revive_by_day')
    ## sum_of_exp_recovery_by_day : acc_id당 각 day에 경험치 복구한 횟수의 합
    sum_of_exp_recovery_by_day = df_merged_origin.groupby(['acc_id','day'])['exp_recovery'].sum().reset_index(name='sum_of_exp_recovery_by_day')
    ## sum_of_fishing_by_day : acc_id당 각 day에 낚시 횟수의 합
    sum_of_fishing_by_day = df_merged_origin.groupby(['acc_id','day'])['fishing'].sum().reset_index(name='sum_of_fishing_by_day')
    ## sum_of_private_shop_by_day : acc_id당 각 day에 개인상점 운영 시간의 합
    sum_of_private_shop_by_day = df_merged_origin.groupby(['acc_id','day'])['private_shop'].sum().reset_index(name='sum_of_private_shop_by_day')
    ## num_of_game_money_change_by_day : acc_id당 각 day에 아데나 변동 횟수의 합
    num_of_game_money_change_by_day = (df_merged_origin.groupby(['acc_id','day'])['game_money_change'].nunique()-1).reset_index(name='num_of_game_money_change_by_day')
    ## sum_of_enchant_count_by_day : acc_id당 각 day에 7 레벨 이상 아이템 인첸트 시도 횟수의 합
    sum_of_enchant_count_by_day = df_merged_origin.groupby(['acc_id','day'])['enchant_count'].sum().reset_index(name='sum_of_enchant_count_by_day')

    tmp_df = pd.merge(id_day_base, num_of_characters_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, num_of_days_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, num_of_servers_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_playtime_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_npckill_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_soloexp_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_partyexp_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_questexp_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_boss_days_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_death_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_revive_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_exp_recovery_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_fishing_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_private_shop_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, num_of_game_money_change_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_enchant_count_by_day, on=['acc_id', 'day'], how='left').fillna(0)

    variable_list = ['num_of_characters',
    'num_of_days',
    'num_of_servers',
    'sum_of_playtime',
    'sum_of_npckill',
    'sum_of_soloexp',
    'sum_of_partyexp',
    'sum_of_questexp',
    'sum_of_boss_days',
    'sum_of_death',
    'sum_of_revive',
    'sum_of_exp_recovery',
    'sum_of_fishing',
    'sum_of_private_shop',
    'num_of_game_money_change',
    'sum_of_enchant_count']

    for variable_name in variable_list:
        variable_day = variable_name+'_by_day'
        df_temp_acti = tmp_df.pivot_table(variable_day, 'acc_id', 'day').reset_index()
        list_temp = ['acc_id']
        for i in range(1,day_width+1):
            list_temp.append('day_%d_%s'%(i,variable_name))
        df_temp_acti.columns = list_temp
        df_acti = pd.merge(df_acti, df_temp_acti, on='acc_id', how='left')

    ## act_first_day : acc_id 당 처음으로 활동한 날
    act_first_day = df_acti_origin.groupby(['acc_id'])['day'].min().reset_index(name='act_first_day')
    ## act_last_day : acc_id 당 마지막으로 활동한 날
    act_last_day = df_acti_origin.groupby(['acc_id'])['day'].max().reset_index(name='act_last_day')
    ## act_first_last_diff : acc_id 당 처음과 마지막으로 활동한 날 수의 차이
    acti_diff = act_last_day['act_last_day'] - act_first_day['act_first_day']
    acti_diff = pd.Series(acti_diff).reset_index(name='act_first_last_diff')

    df_acti_day = pd.concat((acti_diff,act_first_day),axis=1).iloc[:,1:]
    df_acti = pd.merge(df_acti, df_acti_day, on=['acc_id'], how='left').fillna(0)


    ### 2. combat
    df_merged_origin = df_comb_origin

    train_activity = df_acti_origin.sort_values(by=['acc_id', 'day']).reset_index(drop=True) # 주 활동
    train_combat = df_comb_origin.sort_values(by=['acc_id', 'day']).reset_index(drop=True) # 전투
    train_pledge = df_pled_origin.sort_values(by=['acc_id', 'day']).reset_index(drop=True) # 혈맹

    ######################### 주어진 combat data에서의 전투활동변수 #########################
    # pledge_cnt          : 혈맹간 전투에 참여한 횟수
    # random_attacker_cnt : 본인이 막피 공격을 행한 횟수
    # random_defender_cnt : 막피 공격자로부터 공격을 받은 횟수
    # temp_cnt            : 단발성 전투 횟수
    # same_pledge_cnt     : 동일 혈맹원 간의 전투 횟수
    # etc_cnt             : 기타 전투 횟수
    # num_opponent        : 전투 상대 캐릭터 수
    #######################################################################################

    ## combat_days : 전투활동일수. combat data상에 기록이 있는 경우의 day를 count함
    ## combat_active_days : 전투활성일수. combat data상에서 전투활동변수가 0이 아닐 때의 day를 count함
    ## pledge_days : 혈맹활동일수
    combat_days = train_combat.groupby(['acc_id'])['day'].nunique().reset_index(name='combat_days')
    combat_active_yes_index = sorted(set(train_combat.index)-set(train_combat[(train_combat['pledge_cnt']==0)&(train_combat['random_attacker_cnt']==0)&(train_combat['random_defender_cnt']==0)&(train_combat['temp_cnt']==0)&(train_combat['same_pledge_cnt']==0)&(train_combat['etc_cnt']==0)&(train_combat['num_opponent']==0)].index))
    combat_active_days = train_combat.iloc[combat_active_yes_index].groupby('acc_id')['day'].nunique().reset_index(name='combat_active_days')

    pledge_days = train_pledge.groupby(['acc_id'])['day'].nunique().reset_index(name='pledge_days')
    acting_days = pd.merge(combat_days, combat_active_days, on='acc_id', how='left')
    acting_days = pd.merge(acting_days, pledge_days, on='acc_id', how='left')
    acting_days = acting_days.fillna(0)

    acting_days['combat_days'] = acting_days['combat_days'].astype(int)
    acting_days['combat_active_days'] = acting_days['combat_active_days'].astype(int)
    acting_days['pledge_days'] = acting_days['pledge_days'].astype(int)

    ## combat_active_diff : 전투활동일과 전투활성일 차이
    acting_days['combat_active_diff'] = acting_days['combat_days'] - acting_days['combat_active_days']

    ## combat_active_days_rate : 전투활성일이 전투활동일 중 차지하는 비율
    acting_days['combat_active_days_rate'] = acting_days['combat_active_days']/acting_days['combat_days']

    ## combat_first_day : combat data상 최초 접속일
    ## combat_last_day : combat data상 마지막 접속일
    combat_first_day = train_combat.groupby(['acc_id'])['day'].min().reset_index(name='combat_first_day')
    combat_last_day = train_combat.groupby(['acc_id'])['day'].max().reset_index(name='combat_last_day')
    acting_days = pd.merge(acting_days, combat_first_day, on='acc_id', how='left')
    acting_days = pd.merge(acting_days, combat_last_day, on='acc_id', how='left')

    ## combat_first_last_diff_day : combat data상 최초 접속일과 마지막 접속일 차이
    acting_days['combat_first_last_diff_day'] = acting_days['combat_last_day'] - acting_days['combat_first_day']

    ## combat_active_first_day : 전투 활성일 기준 최초로 한 날
    ## combat_active_last_day : 전투 활성일 기준 마지막으로 한 날
    combat_active_yes = train_combat.iloc[combat_active_yes_index]
    combat_active_first_day = combat_active_yes.groupby(['acc_id'])['day'].min().reset_index(name='combat_active_first_day')
    combat_active_last_day = combat_active_yes.groupby(['acc_id'])['day'].max().reset_index(name='combat_active_last_day')
    acting_days = pd.merge(acting_days, combat_active_first_day, on='acc_id', how='left')
    acting_days = pd.merge(acting_days, combat_active_last_day, on='acc_id', how='left')

    ## combat_active_first_last_diff : 전투 활성일 기준 최초로 한 날과 마지막으로 한 날의 차이
    acting_days['combat_active_first_last_diff'] = acting_days['combat_active_last_day'] - acting_days['combat_active_first_day']
    acting_days = acting_days.fillna(0)

    ## combat_last_day 제거 (다 마지막날에는 접속했음)
    acting_days.drop(['combat_last_day'], 1, inplace=True)

    ## combat_server_cnt : 전투를 한 서버 개수
    server_count_by_user = train_combat.groupby(['acc_id'])['server'].nunique().reset_index(name='combat_server_cnt')
    df_by_user = pd.merge(acting_days, server_count_by_user, on='acc_id')

    ## combat_server_cnt_by_day : 평균적인 전투 서버 개수
    df_by_user['combat_server_cnt_by_day'] = train_combat.groupby(['day', 'acc_id'])['server'].nunique().reset_index(name='combat_server_cnt_by_day').groupby(['acc_id'])['combat_server_cnt_by_day'].mean().reset_index(name='combat_server_cnt_by_day')['combat_server_cnt_by_day']

    ## class_cnt : acc_id당 보유한 class의 수
    class_count_by_user = train_combat.groupby(['acc_id'])['class'].nunique().reset_index(name='class_cnt')
    df_by_user = pd.merge(df_by_user, class_count_by_user, on='acc_id')

    ## char_cnt : acc_id당 보유한 캐릭터의 수
    id_count_by_user = train_combat.groupby(['acc_id'])['char_id'].nunique().reset_index().rename(columns={'char_id':'char_cnt'})
    df_by_user = pd.merge(df_by_user, id_count_by_user, on='acc_id')

    ## total_min_level : acc_id가 보유한 캐릭터 중 최저 레벨
    total_min_level = train_combat.groupby(['acc_id'], as_index=False)['level'].min().rename(columns={'level':'total_min_level'})

    ## total_max_level : acc_id가 보유한 캐릭터 중 최고 레벨
    total_max_level = train_combat.groupby(['acc_id'], as_index=False)['level'].max().rename(columns={'level':'total_max_level'})
    df_by_user = pd.merge(df_by_user, total_min_level, on='acc_id', how='left')
    df_by_user = pd.merge(df_by_user, total_max_level, on='acc_id', how='left')

    ##### acc_id의 캐릭터별 레벨 down, keep, up 알아보기 #####
    # acc_id의 캐릭터당 최저 레벨
    min_level = train_combat.groupby(['acc_id', 'char_id', 'class'], as_index=False)['level'].min().rename(columns={'level':'min_level'})

    # acc_id의 캐릭터당 최고 레벨
    max_level = train_combat.groupby(['acc_id', 'char_id', 'class'], as_index=False)['level'].max().rename(columns={'level':'max_level'})
    min_max_level = pd.merge(min_level, max_level, on=['acc_id', 'char_id', 'class'])
    min_max_level['min_max_diff'] = min_max_level['max_level']- min_max_level['min_level']

    # 하루의 맥스 레벨
    max_level_by_day = train_combat.groupby(['day', 'acc_id', 'char_id', 'class'], as_index=False)['level'].max()

    # 최초접속 day
    first_day = train_combat.groupby(['acc_id', 'char_id', 'class'])['day'].min().reset_index(name='first_day')
    first_day_level = pd.merge(first_day, max_level_by_day[['day', 'acc_id', 'char_id', 'class', 'level']].drop_duplicates(), left_on=['acc_id', 'char_id', 'class', 'first_day'], right_on=['acc_id', 'char_id', 'class', 'day'], how='left').drop(['day'], 1).rename(columns={'level':'first_day_level'})

    # 마지막 접속 day
    last_day = train_combat.groupby(['acc_id', 'char_id', 'class'])['day'].max().reset_index(name='last_day')
    last_day_level = pd.merge(last_day, max_level_by_day[['day', 'acc_id', 'char_id', 'class', 'level']].drop_duplicates(), left_on=['acc_id', 'char_id', 'class', 'last_day'], right_on=['acc_id', 'char_id', 'class', 'day'], how='left').drop(['day'], 1).rename(columns={'level':'last_day_level'})
    min_max_level_by_day = pd.merge(first_day_level, last_day_level, on=['acc_id', 'char_id', 'class'])
    min_max_level_by_day['min_max_diff'] = min_max_level_by_day['last_day_level'] - min_max_level_by_day['first_day_level']

    # 가지고 있는 캐릭터 중에 레벨 down 유지 up 개수 세어보기
    level_down_char_cnt = min_max_level_by_day[min_max_level_by_day['min_max_diff']<0].groupby(['acc_id'])['char_id'].nunique().reset_index(name='level_down_char_cnt')
    level_keep_char_cnt = min_max_level_by_day[min_max_level_by_day['min_max_diff']==0].groupby(['acc_id'])['char_id'].nunique().reset_index(name='level_keep_char_cnt')
    level_up_char_cnt = min_max_level_by_day[min_max_level_by_day['min_max_diff']>0].groupby(['acc_id'])['char_id'].nunique().reset_index(name='level_up_char_cnt')

    df_by_user = pd.merge(df_by_user, level_down_char_cnt, on=['acc_id'], how='left')
    df_by_user = pd.merge(df_by_user, level_keep_char_cnt, on=['acc_id'], how='left')
    df_by_user = pd.merge(df_by_user, level_up_char_cnt, on=['acc_id'], how='left')

    ## level_down_char_cnt : acc_id당 레벨 down을 한 캐릭터의 개수
    df_by_user['level_down_char_cnt'] = df_by_user['level_down_char_cnt'].fillna(0).astype(int)

    ## level_keep_char_cnt : acc_id당 레벨을 유지한 캐릭터의 개수
    df_by_user['level_keep_char_cnt'] = df_by_user['level_keep_char_cnt'].fillna(0).astype(int)

    ## level_up_char_cnt : acc_id당 레벨 up을 한 캐릭터의 개수
    df_by_user['level_up_char_cnt'] = df_by_user['level_up_char_cnt'].fillna(0).astype(int)

    ## level_down_rate : acc_id당 레벨 down을 한 캐릭터의 비율
    df_by_user = df_by_user.assign(level_down_rate = df_by_user['level_down_char_cnt']/df_by_user['char_cnt'])

    ## level_keep_rate : acc_id당 레벨을 유지한 캐릭터의 비율
    df_by_user = df_by_user.assign(level_keep_rate = df_by_user['level_keep_char_cnt']/df_by_user['char_cnt'])

    ## level_up_rate : acc_id당 레벨 up을 한 캐릭터의 비율
    df_by_user = df_by_user.assign(level_up_rate = df_by_user['level_up_char_cnt']/df_by_user['char_cnt'])

    ## sp_cls_cnt : acc_id가 보유한 캐릭터 중 특별한 클래스(마법사, 요정, 다크엘프)인 캐릭터 개수
    #※특별한 클래스 : 전투 기록에서 두각을 나타내는 클래스들
    sp_cls_cnt = train_combat[(train_combat['class']==3)|(train_combat['class']==2)|(train_combat['class']==4)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='sp_cls_cnt')
    df_by_user = pd.merge(df_by_user, sp_cls_cnt, on='acc_id', how='left')
    df_by_user['sp_cls_cnt'] = df_by_user['sp_cls_cnt'].fillna(0).astype(int)

    ## sp_cls_rate : acc_id가 보유한 캐릭터 중 특별한 클래스(마법사, 요정, 다크엘프)인 캐릭터가 차지하는 비율
    df_by_user['sp_cls_rate'] = df_by_user['sp_cls_cnt'] / df_by_user['char_cnt']

    ## magic_cnt, magic_rate : acc_id가 보유한 캐릭터 중 마법사인 캐릭터 (개수, 비율)
    magic_cnt = train_combat[(train_combat['class']==3)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='magic_cnt')
    df_by_user = pd.merge(df_by_user, magic_cnt, on='acc_id', how='left')
    df_by_user['magic_cnt'] = df_by_user['magic_cnt'].fillna(0).astype(int)
    df_by_user['magic_rate'] = df_by_user['magic_cnt'] / df_by_user['char_cnt']

    ## fairy_cnt, fairy_rate : acc_id가 보유한 캐릭터 중 요정인 캐릭터 (개수, 비율)
    fairy_cnt = train_combat[(train_combat['class']==2)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='fairy_cnt')
    df_by_user = pd.merge(df_by_user, fairy_cnt, on='acc_id', how='left')
    df_by_user['fairy_cnt'] = df_by_user['fairy_cnt'].fillna(0).astype(int)
    df_by_user['fairy_rate'] = df_by_user['fairy_cnt'] / df_by_user['char_cnt']

    ## dark_cnt, dark_rate : acc_id가 보유한 캐릭터 중 다크엘프인 캐릭터 (개수, 비율)
    dark_cnt = train_combat[(train_combat['class']==4)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='dark_cnt')
    df_by_user = pd.merge(df_by_user, dark_cnt, on='acc_id', how='left')
    df_by_user['dark_cnt'] = df_by_user['dark_cnt'].fillna(0).astype(int)
    df_by_user['dark_rate'] = df_by_user['dark_cnt'] / df_by_user['char_cnt']

    ## king_cnt, king_rate : acc_id가 보유한 캐릭터 중 군주인 캐릭터 (개수, 비율)
    king_cnt = train_combat[(train_combat['class']==0)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='king_cnt')
    df_by_user = pd.merge(df_by_user, king_cnt, on='acc_id', how='left')
    df_by_user['king_cnt'] = df_by_user['king_cnt'].fillna(0).astype(int)
    df_by_user['king_rate'] = df_by_user['king_cnt'] / df_by_user['char_cnt']

    ## chivalry_cnt, chivalry_rate : acc_id가 보유한 캐릭터 중 기사인 캐릭터 (개수, 비율)
    chivalry_cnt = train_combat[(train_combat['class']==1)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='chivalry_cnt')
    df_by_user = pd.merge(df_by_user, chivalry_cnt, on='acc_id', how='left')
    df_by_user['chivalry_cnt'] = df_by_user['chivalry_cnt'].fillna(0).astype(int)
    df_by_user['chivalry_rate'] = df_by_user['chivalry_cnt'] / df_by_user['char_cnt']

    ## dragon_cnt, dragon_rate : acc_id가 보유한 캐릭터 중 용기사인 캐릭터 (개수, 비율)
    dragon_cnt = train_combat[(train_combat['class']==5)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='dragon_cnt')
    df_by_user = pd.merge(df_by_user, dragon_cnt, on='acc_id', how='left')
    df_by_user['dragon_cnt'] = df_by_user['dragon_cnt'].fillna(0).astype(int)
    df_by_user['dragon_rate'] = df_by_user['dragon_cnt'] / df_by_user['char_cnt']

    ## emperor_cnt, emperor_rate : acc_id가 보유한 캐릭터 중 환술사인 캐릭터 (개수, 비율)
    emperor_cnt = train_combat[(train_combat['class']==6)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='emperor_cnt')
    df_by_user = pd.merge(df_by_user, emperor_cnt, on='acc_id', how='left')
    df_by_user['emperor_cnt'] = df_by_user['emperor_cnt'].fillna(0).astype(int)
    df_by_user['emperor_rate'] = df_by_user['emperor_cnt'] / df_by_user['char_cnt']

    ## warrior_cnt, warrior_rate : acc_id가 보유한 캐릭터 중 전사인 캐릭터 (개수, 비율)
    warrior_cnt = train_combat[(train_combat['class']==7)][['acc_id', 'char_id', 'class']].drop_duplicates().groupby('acc_id')['char_id'].nunique().reset_index(name='warrior_cnt')
    df_by_user = pd.merge(df_by_user, warrior_cnt, on='acc_id', how='left')
    df_by_user['warrior_cnt'] = df_by_user['warrior_cnt'].fillna(0).astype(int)
    df_by_user['warrior_rate'] = df_by_user['warrior_cnt'] / df_by_user['char_cnt']

    # pledge_id_count : acc_id당 가입한 혈맹 개수
    pledge_id_count_by_user = train_pledge.groupby(['acc_id'])['pledge_id'].nunique().reset_index().rename(columns={'pledge_id':'pledge_id_count'})
    df_by_user = pd.merge(df_by_user, pledge_id_count_by_user, on='acc_id', how='left')
    df_by_user = df_by_user.fillna(0)
    df_by_user['pledge_id_count'] = df_by_user['pledge_id_count'].astype(int)

    # pledge_combat_active_rate : 혈맹의 전투율. 게임에 접속한 혈맹원 대비 전투에 참여한 혈맹원의 비율
    #                             전투에 참여한 사람(전투에 참여한 혈맹원 수) / 게임에 접속한 사람(게임에 접속한 혈맹원 수)
    df_by_pledge = train_pledge.groupby(['pledge_id'], as_index=False)[['play_char_cnt',
        'combat_char_cnt', 'pledge_combat_cnt', 'random_attacker_cnt',
        'random_defender_cnt', 'same_pledge_cnt', 'temp_cnt', 'etc_cnt',
        'combat_play_time', 'non_combat_play_time']].mean()
    df_by_pledge['pledge_combat_active_rate'] = df_by_pledge['combat_char_cnt']/df_by_pledge['play_char_cnt']

    # combat_active_yn : 전투 혈맹 여부. 혈맹이 전투 위주의 활동을 하는지(1) 혈맹이 친목 위주의 활동을 하는지(0) 구분
    #                    pledge_combat_active_rate가 0이면 친목 혈맹, 0보다 크면 전투 혈맹으로 분류
    df_by_pledge['pledge_combat_active_yn'] = np.where(df_by_pledge['pledge_combat_active_rate']>0, 1, 0)

    ## friend_pledge_cnt : acc_id가 가입한 혈맹 중 친목 위주 혈맹의 개수
    ## combat_pledge_cnt : acc_id가 가입한 혈맹 중 전투 위주 혈맹의 개수
    friend_pledge_cnt_by_user = train_pledge[train_pledge['pledge_id'].isin(df_by_pledge[df_by_pledge['pledge_combat_active_yn']==0]['pledge_id'].unique())].groupby(['acc_id'])['pledge_id'].nunique().reset_index().rename(columns={'pledge_id':'friend_pledge_cnt'})
    combat_pledge_cnt_by_user = train_pledge[train_pledge['pledge_id'].isin(df_by_pledge[df_by_pledge['pledge_combat_active_yn']==1]['pledge_id'].unique())].groupby(['acc_id'])['pledge_id'].nunique().reset_index().rename(columns={'pledge_id':'combat_pledge_cnt'})
    df_by_user = pd.merge(df_by_user, friend_pledge_cnt_by_user, on='acc_id', how='left')
    df_by_user = pd.merge(df_by_user, combat_pledge_cnt_by_user, on='acc_id', how='left')
    df_by_user['friend_pledge_cnt'] = df_by_user['friend_pledge_cnt'].fillna(0).astype(int)
    df_by_user['combat_pledge_cnt'] = df_by_user['combat_pledge_cnt'].fillna(0).astype(int)

    ## friend_pledge_rate : 가입 혈맹 중 친목 위주 혈맹 참여비율
    ## combat_pledge_rate : 가입 혈맹 중 전투 위주 혈맹 참여비율
    df_by_user['friend_pledge_rate'] = df_by_user['friend_pledge_cnt']/df_by_user['pledge_id_count']
    df_by_user['combat_pledge_rate'] = df_by_user['combat_pledge_cnt']/df_by_user['pledge_id_count']
    df_by_user = df_by_user.fillna(0)

    # pledge_id_count 제거
    df_by_user.drop(['pledge_id_count'], 1, inplace=True)

    ## sum_of_전투활동변수 : acc_id당 각 전투활동변수의 합
    df_comb = train_combat.groupby(['acc_id'], as_index=False)[['pledge_cnt', 'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
           'same_pledge_cnt', 'etc_cnt', 'num_opponent']].sum().rename(columns={'pledge_cnt':'sum_of_pledge_cnt',
                                                                                'random_attacker_cnt':'sum_of_random_attacker_cnt',
                                                                               'random_defender_cnt':'sum_of_random_defender_cnt',
                                                                                'temp_cnt':'sum_of_temp_cnt',
                                                                                'same_pledge_cnt':'sum_of_same_pledge_cnt',
                                                                                'etc_cnt':'sum_of_etc_cnt',
                                                                                'num_opponent':'sum_of_num_opponent'})
    df_comb = pd.merge(df_by_user, df_comb, on='acc_id')

    ## mean_of_전투활동변수 : 주어진 combat data에서 acc_id당 각 전투활동변수의 평균 = sum_of_전투활동변수/전투활동일수
    df_comb['mean_of_pledge_cnt'] = df_comb['sum_of_pledge_cnt']/df_comb['combat_days']
    df_comb['mean_of_random_attacker_cnt'] = df_comb['sum_of_random_attacker_cnt']/df_comb['combat_days']
    df_comb['mean_of_random_defender_cnt'] = df_comb['sum_of_random_defender_cnt']/df_comb['combat_days']
    df_comb['mean_of_temp_cnt'] = df_comb['sum_of_temp_cnt']/df_comb['combat_days']
    df_comb['mean_of_same_pledge_cnt'] = df_comb['sum_of_same_pledge_cnt']/df_comb['combat_days']
    df_comb['mean_of_etc_cnt'] = df_comb['sum_of_etc_cnt']/df_comb['combat_days']
    df_comb['mean_of_num_opponent'] = df_comb['sum_of_num_opponent']/df_comb['combat_days']
    df_comb = df_comb.fillna(0)

    ## act_mean_of_전투활동변수 :실질적으로 전투활동을 한 날의 acc_id당 각 전투활동변수의 평균 = sum_of_전투활동/전투활성일수
    df_comb['act_mean_of_pledge_cnt'] = df_comb['sum_of_pledge_cnt']/df_comb['combat_active_days']
    df_comb['act_mean_of_random_attacker_cnt'] = df_comb['sum_of_random_attacker_cnt']/df_comb['combat_active_days']
    df_comb['act_mean_of_random_defender_cnt'] = df_comb['sum_of_random_defender_cnt']/df_comb['combat_active_days']
    df_comb['act_mean_of_temp_cnt'] = df_comb['sum_of_temp_cnt']/df_comb['combat_active_days']
    df_comb['act_mean_of_same_pledge_cnt'] = df_comb['sum_of_same_pledge_cnt']/df_comb['combat_active_days']
    df_comb['act_mean_of_etc_cnt'] = df_comb['sum_of_etc_cnt']/df_comb['combat_active_days']
    df_comb['act_mean_of_num_opponent'] = df_comb['sum_of_num_opponent']/df_comb['combat_active_days']
    df_comb = df_comb.fillna(0)

    ## day_#_sum_of_전투활동변수 : acc_id당 각 day에 활동한 전투활동변수의 합
    df_merged_origin = df_comb_origin

    unique_id =  df_merged_origin['acc_id'].drop_duplicates().sort_values().reset_index(drop=True)
    acc_ids = pd.DataFrame(pd.concat([unique_id]*day_width)).reset_index(drop=True)
    days = pd.Series(np.repeat(np.arange(1, day_width + 1), len(unique_id)))
    acc_ids['day'] = days
    id_day_base = acc_ids.sort_values(['acc_id', 'day']).reset_index(drop=True)

    sum_of_pledge_cnt_by_day = df_merged_origin.groupby(['acc_id','day'])['pledge_cnt'].sum().reset_index(name='sum_of_pledge_cnt_by_day')
    sum_of_random_attacker_cnt_by_day = df_merged_origin.groupby(['acc_id','day'])['random_attacker_cnt'].sum().reset_index(name='sum_of_random_attacker_cnt_by_day')
    sum_of_random_defender_cnt_by_day = df_merged_origin.groupby(['acc_id','day'])['random_defender_cnt'].sum().reset_index(name='sum_of_random_defender_cnt_by_day')
    sum_of_temp_cnt_by_day = df_merged_origin.groupby(['acc_id','day'])['temp_cnt'].sum().reset_index(name='sum_of_temp_cnt_by_day')
    sum_of_same_pledge_cnt_by_day = df_merged_origin.groupby(['acc_id','day'])['same_pledge_cnt'].sum().reset_index(name='sum_of_same_pledge_cnt_by_day')
    sum_of_etc_cnt_by_day = df_merged_origin.groupby(['acc_id','day'])['etc_cnt'].sum().reset_index(name='sum_of_etc_cnt_by_day')
    sum_of_num_opponent_by_day = df_merged_origin.groupby(['acc_id','day'])['num_opponent'].sum().reset_index(name='sum_of_num_opponent_by_day')

    tmp_df = pd.merge(id_day_base, sum_of_pledge_cnt_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_random_attacker_cnt_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_random_defender_cnt_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_temp_cnt_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_same_pledge_cnt_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_etc_cnt_by_day, on=['acc_id', 'day'], how='left').fillna(0)
    tmp_df = pd.merge(tmp_df, sum_of_num_opponent_by_day, on=['acc_id', 'day'], how='left').fillna(0)

    variable_list = ['sum_of_pledge_cnt',
    'sum_of_random_attacker_cnt',
    'sum_of_random_defender_cnt',
    'sum_of_temp_cnt',
    'sum_of_same_pledge_cnt',
    'sum_of_etc_cnt',
    'sum_of_num_opponent']

    for variable_name in variable_list:
        variable_day = variable_name+'_by_day'
        df_temp_comb = tmp_df.pivot_table(variable_day, 'acc_id', 'day').reset_index()
        list_temp = ['acc_id']
        for i in range(1,day_width + 1):
            list_temp.append('day_%d_%s'%(i,variable_name))
        df_temp_comb.columns = list_temp
        df_comb = pd.merge(df_comb, df_temp_comb, on='acc_id', how='left')


    ### 3. trade
    df = df_trad_origin

    ## hour : acc_id가 거래를 진행한 '시각(시)' / minute : acc_id가 거래를 진행한 '시각(분)'
    df['hour'] = df['time'].map(lambda x: int(x[0:2]))
    df['minute'] = df['time'].map(lambda x: int(x[3:5]))
    df['time'] = df['time'].map(lambda x: time(hour = int(x[0:5][0:2]), minute = int(x[0:5][3:5])))

    #print("1. Label과 trade df에 공통으로 들어있는 user id를 추출하여, 이를 기준으로 분석을 진행하자.\n")
    sell_users = df.source_acc_id.unique().tolist()
    buy_users = df.target_acc_id.unique().tolist()

    buy_sell_union = list(set().union(sell_users, buy_users)) ### 병합된 데이터의 총 row가 되는 user id
    buy_sell_intersect = list(set(sell_users).intersection(set(buy_users))) ### Label의 user id 중 Sell, buy 이력이 모두 있는 user id
    only_in_trade_sell = list(set(sell_users) - set(buy_sell_intersect)) ### Label의 user id 중 Sell 이력만 있는 user id
    only_in_trade_buy = list(set(buy_users) - set(buy_sell_intersect))### Label의 user id 중 buy 이력만 있는 user id

    sell_price = df.loc[df['source_acc_id'].isin(sell_users)][['item_type', 'item_price']].groupby('item_type').agg({'item_price':'mean'}).reset_index()
    buy_price = df.loc[df['target_acc_id'].isin(buy_users)][['item_type', 'item_price']].groupby('item_type').agg({'item_price':'mean'}).reset_index()
    sell_price['item_price'][1] = 0
    buy_price['item_price'][1] = 0
    # item_price에 대한 결측치 처리(거래되는 각 item_type의 평균 item_price로 대체!)
    ds_nan = df.loc[(df['source_acc_id'].isin(sell_users))&(df['item_price'].isna())]
    ds_nonan = df.loc[(df['source_acc_id'].isin(sell_users))&(df['item_price'].notna())]
    db_nan = df.loc[(df['target_acc_id'].isin(buy_users))&(df['item_price'].isna())]
    db_nonan = df.loc[(df['target_acc_id'].isin(buy_users))&(df['item_price'].notna())]

    df_list = []
    sell_itemtype_list = sell_price['item_type'].tolist()
    for itemtype in sell_itemtype_list:
        df_selltype = ds_nan.loc[ds_nan['item_type'] == itemtype]
        df_selltype['item_price'] = sell_price.loc[sell_price['item_type'] == itemtype].iloc[0,1]
        df_list.append(df_selltype)
    df_sell = pd.concat([pd.concat(objs = df_list, axis = 0, sort = True), ds_nonan], axis = 0, sort = True)

    df_list = []
    buy_itemtype_list = buy_price['item_type'].tolist()
    for itemtype in buy_itemtype_list:
        df_buytype = db_nan.loc[db_nan['item_type'] == itemtype]
        df_buytype['item_price'] = sell_price.loc[sell_price['item_type'] == itemtype].iloc[0,1]
        df_list.append(df_buytype)
    df_buy = pd.concat([pd.concat(objs = df_list, axis = 0, sort = True), db_nonan], axis = 0, sort = True)

    # 각 user 별 Day 1~28에서의 판매/구매 횟수, 판매/구매량, 판매/구매금액 정보
    ## day_#_sum_of_sell : acc_id 별 각 day당 아이템 '판매' 거래 수의 합
    a1 = df_sell.groupby(['source_acc_id', 'day']).count().reset_index().iloc[:,0:3].rename(columns = {'hour':'sell_day_cnt',
                                                                                                          'source_acc_id':'acc_id'})
    sum_sell_by_day_df = a1.pivot_table('sell_day_cnt', 'acc_id', 'day').reset_index().fillna(0)
    sum_sell_by_day_df.columns = ['acc_id'] + [('day_' + str(i) + '_sum_of_sell') for i in range(1,day_width+1)]

    ## day_#_sum_of_buy : acc_id 별 각 day당 아이템 '구매' 거래 수의 합
    a2 = df_buy.groupby(['target_acc_id', 'day']).count().reset_index().iloc[:,0:3].rename(columns = {'hour':'buy_day_cnt',
                                                                                                              'target_acc_id':'acc_id'})
    sum_buy_by_day_df = a2.pivot_table('buy_day_cnt', 'acc_id', 'day').reset_index().fillna(0)
    sum_buy_by_day_df.columns = ['acc_id'] + [('day_' + str(i) + '_sum_of_buy') for i in range(1,day_width+1)]

    ## day_#_sum_of_buy_item_price : acc_id 별 각 day당 아이템 '구매 가격'의 합
    a3 = df_buy.groupby(['target_acc_id','day'])['item_price'].sum().reset_index(name = 'buy_day_item_price').rename(columns = {'target_acc_id':'acc_id'})
    sum_buy_itempr_by_day_df = a3.pivot_table('buy_day_item_price', 'acc_id', 'day').reset_index().fillna(0)
    sum_buy_itempr_by_day_df.columns = ['acc_id'] + [('day_' + str(i) + '_sum_of_buy_item_price') for i in range(1,day_width +1)]

    ## day_#_sum_of_buy_item_price : acc_id 별 각 day당 아이템 '구매량'의 합
    a4 = df_buy.groupby(['target_acc_id','day'])['item_amount'].sum().reset_index(name = 'buy_day_item_amount').rename(columns = {'target_acc_id':'acc_id'})
    sum_buy_itemamt_by_day_df = a4.pivot_table('buy_day_item_amount', 'acc_id', 'day').reset_index().fillna(0)
    sum_buy_itemamt_by_day_df.columns = ['acc_id'] + [('day_' + str(i) + '_sum_of_buy_item_amount') for i in range(1,day_width+1)]

    ## day_#_sum_of_sell_item_price : acc_id 별 각 day당 아이템 '판매 가격'의 합
    a5 = df_sell.groupby(['source_acc_id','day'])['item_price'].sum().reset_index(name = 'sell_day_item_price').rename(columns = {'source_acc_id':'acc_id'})
    sum_sell_itempr_by_day_df = a5.pivot_table('sell_day_item_price', 'acc_id', 'day').reset_index().fillna(0)
    sum_sell_itempr_by_day_df.columns = ['acc_id'] + [('day_' + str(i) + '_sum_of_sell_item_price') for i in range(1,day_width +1)]

    ## day_#_sum_of_sell_item_amount : acc_id 별 각 day당 아이템 '판매량'의 합
    a6 = df_sell.groupby(['source_acc_id','day'])['item_amount'].sum().reset_index(name = 'sell_day_item_amount').rename(columns = {'source_acc_id':'acc_id'})
    sum_sell_itemamt_by_day_df = a6.pivot_table('sell_day_item_amount', 'acc_id', 'day').reset_index().fillna(0)
    sum_sell_itemamt_by_day_df.columns = ['acc_id'] + [('day_' + str(i) + '_sum_of_sell_item_amount') for i in range(1,day_width +1)]

    trade_df_days = sum_sell_by_day_df.merge(sum_buy_by_day_df, how = 'left', on = 'acc_id')
    trade_df_days = trade_df_days.merge(sum_buy_itempr_by_day_df, how = 'left', on = 'acc_id')
    trade_df_days = trade_df_days.merge(sum_buy_itemamt_by_day_df, how = 'left', on = 'acc_id')
    trade_df_days = trade_df_days.merge(sum_sell_itempr_by_day_df, how = 'left', on = 'acc_id')
    trade_df_days = trade_df_days.merge(sum_sell_itemamt_by_day_df, how = 'left', on = 'acc_id')

    trade_sell = df_sell.loc[df_sell['source_acc_id'].isin(buy_sell_intersect)]
    trade_buy = df_buy.loc[df_buy['target_acc_id'].isin(buy_sell_intersect)]
    trade_sell_only = df_sell.loc[df_sell['source_acc_id'].isin(only_in_trade_sell)]
    trade_buy_only = df_buy.loc[df_buy['target_acc_id'].isin(only_in_trade_buy)]

    # Count 1: 거래 빈도 관련 변수 정보.
    # 1. Sell and buy both
    # (1) Get the number of characters for each user id
    ## num_sell_character : acc_id 별 아이템 판매를 진행한 '캐릭터의 수'
    sell_char_count = trade_sell.groupby(['source_acc_id', 'source_char_id']).count().iloc[:,0].reset_index()
    a1 = sell_char_count.groupby('source_acc_id').count().iloc[:,0].reset_index().rename(columns = {'source_char_id':'num_sell_character'})
    seller_character = sell_char_count.merge(right = a1, how = 'inner', on = 'source_acc_id').rename(columns = {'source_acc_id':'acc_id'})

    ## num_buy_character : acc_id 별 아이템 구매를 진행한 '캐릭터의 수'
    buy_char_count = trade_buy.groupby(['target_acc_id', 'target_char_id']).count().iloc[:,0].reset_index()
    b1 = buy_char_count.groupby('target_acc_id').count().iloc[:,0].reset_index().rename(columns = {'target_char_id':'num_buy_character'})
    buyer_character = buy_char_count.merge(right = b1, how = 'inner', on = 'target_acc_id').rename(columns = {'target_acc_id':'acc_id'})

    buy_sell_char_count = seller_character.merge(buyer_character, how = 'inner', on = 'acc_id')
    buy_sell_char_counts = buy_sell_char_count.sort_values('num_sell_character', ascending = False).drop(['source_char_id', 'target_char_id','day_x','day_y'], axis = 1).drop_duplicates()

    # (2) Get the number of transactions for each user id
    ## num_sell : acc_id 별 총 아이템 '판매 거래 수'
    sell_count = trade_sell.groupby(['source_acc_id']).count().iloc[:,0].reset_index().sort_values('day', ascending = False)
    buy_count = trade_buy.groupby(['target_acc_id']).count().iloc[:,0].reset_index().sort_values('day', ascending = False)
    sell_both_count = sell_count.rename(columns = {'source_acc_id':'acc_id','day':'num_sell'})

    ## num_buy : acc_id 별 총 아이템 '구매 거래 수'
    buy_both_count = buy_count.rename(columns = {'target_acc_id':'acc_id','day':'num_buy'})
    buy_sell_count = sell_both_count.merge(buy_both_count, how = 'inner', on = 'acc_id')

    # 2. Sell only
    # (1) Get the number of characters for each user id
    sell_char_count = trade_sell_only.groupby(['source_acc_id', 'source_char_id']).count().iloc[:,0].reset_index()
    a = sell_char_count.groupby('source_acc_id').count().iloc[:,0].reset_index().rename(columns = {'source_char_id':'num_sell_character'})
    seller_character = sell_char_count.merge(right = a, how = 'inner', on = 'source_acc_id').rename(columns = {'source_acc_id':'acc_id'})
    seller_character['num_buy_character'] = 0
    buy_sell_char_counts = seller_character.sort_values('num_sell_character', ascending = False).drop(['source_char_id','day'], axis = 1).drop_duplicates()

    # (2) Get the number of transactions for each user id
    sell_count = trade_sell_only.groupby(['source_acc_id']).count().iloc[:,0].reset_index().sort_values('day', ascending = False)
    sell_only_count = sell_count.rename(columns = {'source_acc_id':'acc_id','day':'num_sell'})
    sell_only_count['num_buy'] = 0

    # 3. Sell only
    # (1) Get the number of characters for each user id
    buy_char_count = trade_buy_only.groupby(['target_acc_id', 'target_char_id']).count().iloc[:,0].reset_index()
    a = buy_char_count.groupby('target_acc_id').count().iloc[:,0].reset_index().rename(columns = {'target_char_id':'num_buy_character'})
    buyer_character = buy_char_count.merge(right = a, how = 'inner', on = 'target_acc_id').rename(columns = {'target_acc_id':'acc_id'})
    buyer_character['num_sell_character'] = 0
    buy_sell_char_counts = buyer_character.sort_values('num_buy_character', ascending = False).drop(['target_char_id','day'], axis = 1).drop_duplicates()

    # (2) Get the number of transactions for each user id
    buy_count = trade_buy_only.groupby(['target_acc_id']).count().iloc[:,0].reset_index().sort_values('day', ascending = False)
    buy_only_count = buy_count.rename(columns = {'target_acc_id':'acc_id','day':'num_buy'})
    buy_only_count['num_sell'] = 0

    # 4. Merge three results and label
    count_result1 = pd.concat([buy_sell_count, sell_only_count, buy_only_count], axis = 0, sort = True)

    # Count 2: day 별 item type, amount 관련 변수 정보.(By for loop)
    num_server1 = []
    common_item1 = []
    sell_time = []
    sell_type = []
    num_sell_day = []
    last_sell_day = []
    last_sell_item_type = []
    last_sell_item_amount = []
    last_sell_item_price = []
    sell_list = sorted(df['source_acc_id'].unique().tolist())
    for i, user in enumerate(sell_list):
        num_server1.append(len(Counter(df_sell.loc[df_sell['source_acc_id'] == user]['server'])))
        common_item1.append(Counter(df_sell.loc[df_sell['source_acc_id'] == user]['item_type']).most_common(1)[0][0])
        sell_time.append(Counter(df_sell.loc[df_sell['source_acc_id'] == user]['hour']).most_common(1)[0][0])
        sell_type.append(Counter(df_sell.loc[df_sell['source_acc_id'] == user]['type']).most_common(1)[0][0])
        sell_user = df_sell[['source_acc_id','day', 'item_type', 'item_amount', 'item_price']][df_sell['source_acc_id'] == user]
        user_sell_day = sell_user['day'].unique().tolist()

        num_day = len(user_sell_day)
        num_sell_day.append(num_day)


        last_day = user_sell_day[(num_day - 1)]
        last_sell_day.append(last_day)


        last_type = sell_user.loc[sell_user['day'] == last_day]['item_type'].values[0]
        last_amount = sell_user.loc[sell_user['day'] == last_day]['item_amount'].values[0]
        last_price = sell_user.loc[sell_user['day'] == last_day]['item_price'].values[0]
        last_sell_item_type.append(last_type)
        last_sell_item_amount.append(last_amount)
        last_sell_item_price.append(last_price)


        ## num_sell_server : acc_id 별 판매를 진행한 서버의 개수 -> num_trade_server로 통합
        ## common_item_sell : acc_id 별 판매를 가장 많이 한 아이템의 type(categorical)
        ## sell_time : acc_id 별 판매를 가장 많이 진행한 시각(단위 : 시간)(categorical)
        ## sell_type : acc_id 별 판매를 진행한 거래의 type(교환창 거래 or 상점 거래)
        ## num_sell_day : acc_id 별 판매를 진행한 일 수
        ## last_sell_day : acc_id 별 Sell을 진행한 day 중 last day
        ## last_sell_item_type : acc_id 별 마지막 day에 판매한 아이템의 type(categorical)
        ## last_sell_item_amount : acc_id 별 마지막 day에 판매한 아이템의 amount
        ## last_sell_item_price : acc_id 별 마지막 day에 판매한 아이템의 price

    sell_df = pd.DataFrame({'acc_id':sell_list,
                            'num_sell_server':num_server1,
                            'common_item_sell':common_item1,
                            'sell_time':sell_time,
                            'sell_type':sell_type,
                            'num_sell_day':num_sell_day,
                            'last_sell_day':last_sell_day,
                            'last_sell_item_type':last_sell_item_type,
                            'last_sell_item_amount':last_sell_item_amount,
                            'last_sell_item_price':last_sell_item_price})

    num_server2 = []
    common_item2 = []
    buy_time = []
    buy_type = []
    num_buy_day = []
    last_buy_day = []
    last_buy_item_type = []
    last_buy_item_amount = []
    last_buy_item_price = []
    buy_list = sorted(df['target_acc_id'].unique().tolist())
    for j, user in enumerate(buy_list):
        num_server2.append(len(Counter(df_buy.loc[df_buy['target_acc_id'] == user]['server'])))
        common_item2.append(Counter(df_buy.loc[df_buy['target_acc_id'] == user]['item_type']).most_common(1)[0][0])
        buy_time.append(Counter(df_buy.loc[df_buy['target_acc_id'] == user]['hour']).most_common(1)[0][0])
        buy_type.append(Counter(df_buy.loc[df_buy['target_acc_id'] == user]['type']).most_common(1)[0][0])
        buy_user = df_buy[['target_acc_id','day', 'item_type', 'item_amount', 'item_price']][df_buy['target_acc_id'] == user]
        user_buy_day = buy_user['day'].unique().tolist()

        num_day = len(user_buy_day)
        num_buy_day.append(num_day)

        last_day = user_buy_day[(num_day - 1)]
        last_buy_day.append(last_day) # Sell을 진행한 day 중 last day

        last_type = buy_user.loc[buy_user['day'] == last_day]['item_type'].values[0]
        last_amount = buy_user.loc[buy_user['day'] == last_day]['item_amount'].values[0]
        last_price = buy_user.loc[buy_user['day'] == last_day]['item_price'].values[0]
        last_buy_item_type.append(last_type)
        last_buy_item_amount.append(last_amount)
        last_buy_item_price.append(last_price)

        ## num_buy_server : acc_id 별 구매를 진행한 서버의 개수 -> num_trade_server로 통합
        ## common_item_buy : acc_id 별 판매를 가장 많이 한 아이템의 type(categorical)
        ## buy_time : acc_id 별 판매를 가장 많이 진행한 시각(단위 : 시간)(categorical)
        ## buy_type : acc_id 별 판매를 진행한 거래의 type(교환창 거래 or 상점 거래)
        ## num_buy_day : acc_id 별 판매를 진행한 일 수
        ## last_buy_day : acc_id 별 Sell을 진행한 day 중 last day
        ## last_buy_item_type : acc_id 별 마지막 day에 판매한 아이템의 type(categorical)
        ## last_buy_item_amount : acc_id 별 마지막 day에 판매한 아이템의 amount
        ## last_buy_item_price : acc_id 별 마지막 day에 판매한 아이템의 price
    buy_df = pd.DataFrame({'acc_id':buy_list,
                           'num_buy_server':num_server2,
                           'common_item_buy':common_item2,
                           'buy_time':buy_time,
                           'buy_type':buy_type,
                           'num_buy_day':num_buy_day,
                           'last_buy_day':last_buy_day,
                           'last_buy_item_type':last_buy_item_type,
                           'last_buy_item_amount':last_buy_item_amount,
                           'last_buy_item_price':last_buy_item_price})

    sell_common = sell_df.loc[sell_df['acc_id'].isin(buy_sell_intersect)]
    buy_common = buy_df.loc[buy_df['acc_id'].isin(buy_sell_intersect)]
    trade_both = sell_common.merge(buy_common, how = 'outer', on = 'acc_id')
    sell_only = sell_df.loc[sell_df['acc_id'].isin(only_in_trade_sell)]
    buy_only = buy_df.loc[buy_df['acc_id'].isin(only_in_trade_buy)]

    ## num_trade_server : acc_id 별 '거래'를 진행한 서버의 개수 = max(num_sell_server, num_buy_server)
    trade_both['num_trade_server'] = trade_both[['num_sell_server','num_buy_server']].apply(lambda x: max(x), axis = 1)
    trade_both = trade_both.drop(['num_sell_server', 'num_buy_server'], axis=1)
    sell_only = sell_only.rename(columns = {'num_sell_server':'num_trade_server'})
    buy_only = buy_only.rename(columns = {'num_buy_server':'num_trade_server'})

    count_result2 = pd.concat([trade_both, sell_only, buy_only], axis = 0, sort = True).drop_duplicates()

    # Mean 관련 변수 정보.(item_amount, item_price)

    ## sell_item_amount : acc_id 별 아이템 판매량의 평균
    sell_itemamt = df_sell[['source_acc_id','item_amount']].groupby('source_acc_id').agg({'item_amount':'mean'}).reset_index().rename(columns = {'source_acc_id':'acc_id',
                                                                                                                                                    'item_amount':'sell_item_amount'})
    ## buy_item_amount : acc_id 별 아이템 구매량의 평균
    buy_itemamt = df_buy[['target_acc_id','item_amount']].groupby('target_acc_id').agg({'item_amount':'mean'}).reset_index().rename(columns = {'target_acc_id':'acc_id',
                                                                                                                                                      'item_amount':'buy_item_amount'})
    sell_only_itemamt = sell_itemamt.loc[sell_itemamt['acc_id'].isin(only_in_trade_sell)]
    buy_only_itemamt = buy_itemamt.loc[buy_itemamt['acc_id'].isin(only_in_trade_buy)]
    sell2 = sell_itemamt.loc[sell_itemamt['acc_id'].isin(buy_sell_intersect)]
    buy2 = buy_itemamt.loc[buy_itemamt['acc_id'].isin(buy_sell_intersect)]
    sell_buy_itemamt = sell2.merge(buy2, how = 'outer', on = 'acc_id')

    itemamt_result = pd.concat([sell_buy_itemamt, sell_only_itemamt, buy_only_itemamt], axis = 0, sort = True).drop_duplicates()
    ## sell_item_amount : acc_id 별 아이템 판매 가격의 평균
    sell_itempr = df[['source_acc_id','item_price']].groupby('source_acc_id').agg({'item_price':'mean'}).reset_index().rename(columns = {'source_acc_id':'acc_id',
                                                                                                                                                         'item_price':'sell_item_price'})
    ## buy_item_price : acc_id 별 아이템 구매 가격의 평균
    buy_itempr = df[['target_acc_id','item_price']].groupby('target_acc_id').agg({'item_price':'mean'}).reset_index().rename(columns = {'target_acc_id':'acc_id',
                                                                                                                                                          'item_price':'buy_item_price'})
    sell_only_itempr = sell_itempr.loc[sell_itempr['acc_id'].isin(only_in_trade_sell)]
    buy_only_itempr = buy_itempr.loc[buy_itempr['acc_id'].isin(only_in_trade_buy)]
    sell2 = sell_itempr.loc[sell_itempr['acc_id'].isin(buy_sell_intersect)]
    buy2 = buy_itempr.loc[buy_itempr['acc_id'].isin(buy_sell_intersect)]
    sell_buy_itempr = sell2.merge(buy2, how = 'outer', on = 'acc_id')

    itempr_result = pd.concat([sell_buy_itempr, sell_only_itempr, buy_only_itempr], axis = 0, sort = True).drop_duplicates()
    mean_result = itemamt_result.merge(itempr_result, on = 'acc_id')

    df_final = count_result1.merge(count_result2, on = 'acc_id').merge(mean_result, on = 'acc_id').merge(trade_df_days, how = 'left', on = 'acc_id')

    # 결측치 처리 : time, type의 경우 '0'이 의미를 가지는 값이기 때문에 '미관측'을 나타내기 위하여 -1로 결측값 처리
    df_final['sell_time'] = df_final['sell_time'].fillna(-1)
    df_final['buy_time'] = df_final['buy_time'].fillna(-1)
    df_final['sell_type'] = df_final['sell_type'].fillna(-1)
    df_final['buy_type'] = df_final['buy_type'].fillna(-1)

    # category 변수들의 data type을 'category'로 변경
    df_final['common_item_sell'] = df_final['common_item_sell'].astype('category')
    df_final['common_item_buy'] = df_final['common_item_buy'].astype('category')
    df_final['sell_type'] = df_final['sell_type'].astype('category')
    df_final['buy_type'] = df_final['buy_type'].astype('category')
    df_final['last_sell_item_type'] = df_final['last_sell_item_type'].astype('category')
    df_final['last_buy_item_type'] = df_final['last_buy_item_type'].astype('category')

    df_trad = df_final


    ### 4. payment
    train_payment = df_paym_origin
    ## pay_day_cnt : 결제 이벤트 횟수
    dayCnt = pd.DataFrame(train_payment.groupby('acc_id').count()['day']).reset_index()
    dayCnt.columns = ['acc_id','pay_day_cnt']
    # sum_amount_spent : id별 결제 총 금액
    avgSpent = pd.DataFrame(train_payment.groupby('acc_id').sum()['amount_spent']).reset_index()
    avgSpent.columns = ['acc_id', 'sum_amount_spent']

    df_paym = pd.merge(dayCnt,avgSpent)

    #payment_day
    df_merged_origin = df_paym_origin

    unique_id =  df_merged_origin['acc_id'].drop_duplicates().sort_values().reset_index(drop=True)
    acc_ids = pd.DataFrame(pd.concat([unique_id]*day_width)).reset_index(drop=True)
    days = pd.Series(np.repeat(np.arange(1, day_width+1), len(unique_id)))
    acc_ids['day'] = days
    id_day_base = acc_ids.sort_values(['acc_id', 'day']).reset_index(drop=True)

    sum_of_amount_spent_by_day = df_merged_origin.groupby(['acc_id','day'])['amount_spent'].sum().reset_index(name='sum_of_amount_spent_by_day')

    tmp_df = pd.merge(id_day_base, sum_of_amount_spent_by_day, on=['acc_id', 'day'], how='left').fillna(0)

    variable_list = ['sum_of_amount_spent']
    ## day_%d_sum_of_amount_spent : day별 결재 금액
    for variable_name in variable_list:
        variable_day = variable_name+'_by_day'
        df_temp_paym = tmp_df.pivot_table(variable_day, 'acc_id', 'day').reset_index()
        list_temp = ['acc_id']
        for i in range(1,day_width + 1):
            list_temp.append('day_%d_%s'%(i,variable_name))
        df_temp_paym.columns = list_temp
        df_paym = pd.merge(df_paym, df_temp_paym, on='acc_id', how='left')


    ### 5. pledge
    test1_pledge = df_pled_origin
    test1_combat = df_comb_origin
    test1_activity = df_acti_origin

    test1_pledge_arranged = test1_pledge.sort_values(by=['day','acc_id','pledge_id','char_id','server'], axis=0)

    df3 = test1_pledge_arranged.iloc[:,0:5]
    test1_combat1 = test1_combat[['day','acc_id','char_id','server','pledge_cnt']]

    pledge_play = df3.merge(test1_combat1,how='left',on = ["day","acc_id","char_id","server"])
    pledge_play2 = pledge_play[pledge_play['pledge_cnt'] != 0]

    df4 = pledge_play2[['day','acc_id','char_id']].drop_duplicates().groupby(['acc_id','day']).count().reset_index()

    aaa = test1_activity['acc_id'].drop_duplicates().sort_values().reset_index(drop=True)
    acc_ids = pd.DataFrame(pd.concat([aaa]*day_width)).reset_index(drop=True)
    days = pd.Series(np.repeat(np.arange(1, day_width + 1), len(aaa)))
    acc_ids['day'] = days
    id_day_base2 = acc_ids.sort_values(['acc_id', 'day']).reset_index(drop=True)
    ## day_#_combat_char : acc_id당 각 day에 혈맹 전투에 참여한 캐릭터 수
    df5 = id_day_base2.merge(df4,on=['day','acc_id'],how='left').fillna(0).astype(int).rename(columns = {'char_id':'pledge_combat_char'})

    df6 = df5.pivot_table('pledge_combat_char','acc_id','day')

    temperate = []
    for i in range(day_width):
        temperate.append('day_%d_pledge_combat_char' %(i+1))

    df6.columns = temperate

    test1_pledge_combat_char_byday = df6.reset_index() # 오름차순 acc_id

    week1 = pledge_play2[['day','acc_id']][pledge_play2['day'] < 8].drop_duplicates().groupby(['acc_id']).count().reset_index().rename(columns = {'day':'week1_pledge_combat_day'})
    week2 = pledge_play2[['day','acc_id']][(pledge_play2['day'] >= 8) & (pledge_play2['day'] < 15)].drop_duplicates().groupby(['acc_id']).count().reset_index().rename(columns = {'day':'week2_pledge_combat_day'})
    week3 = pledge_play2[['day','acc_id']][(pledge_play2['day'] >= 15) & (pledge_play2['day'] < 22)].drop_duplicates().groupby(['acc_id']).count().reset_index().rename(columns = {'day':'week3_pledge_combat_day'})
    week4 = pledge_play2[['day','acc_id']][pledge_play2['day']>=22 ].drop_duplicates().groupby(['acc_id']).count().reset_index().rename(columns = {'day':'week4_pledge_combat_day'})

    acc_id = pd.DataFrame({'acc_id':aaa})

    ## week#_pledge_combat_day : acc_id당 주차별 혈맹 전투를 한 날짜 수
    week1_pledge_combat_day = acc_id.merge(week1,on=['acc_id'],how='left').fillna(0).astype(int)
    week2_pledge_combat_day = acc_id.merge(week2,on=['acc_id'],how='left').fillna(0).astype(int)
    week3_pledge_combat_day = acc_id.merge(week3,on=['acc_id'],how='left').fillna(0).astype(int)
    week4_pledge_combat_day = acc_id.merge(week4,on=['acc_id'],how='left').fillna(0).astype(int)

    test1_pledge_combat_day_byweek = pd.concat([week1_pledge_combat_day,week2_pledge_combat_day,week3_pledge_combat_day,week4_pledge_combat_day],1).iloc[:,[0,1,3,5,7]]

    df_pled = pd.merge(test1_pledge_combat_char_byday,test1_pledge_combat_day_byweek,on=['acc_id'])

    ## play_char_sum : acc_id의 각 소속 혈맹에서 관찰 기간 동안 활동한 게임접속 캐릭터 수 합의 평균
    ## combat_char_sum : acc_id의 각 소속 혈맹에서 관찰 기간 동안 전투 활동을 한 캐릭터 수 합의 평균
    ## pledge_combat_sum : acc_id의 각 소속 혈맹에서 관찰 기간 동안 발생한 혈맹간 전투 횟수 합의 평균
    ## random_attacker_sum : acc_id의 각 소속 혈맹원중 관찰 기간 동안 막피 전투를 행한 횟수 합의 평균
    ## random_defender_sum : acc_id의 각 소속 혈맹원중 관찰 기간 막피로 피해를 받은 횟수 합의 평균
    ## same_pledge_sum : acc_id의 각 소속 혈맹에서 동일 혈맹원 간 발생한 전투 횟수 합의 평균
    ## temp_sum : aacc_id의 각 소속 혈맹원중 관찰 기간 막피로 피해를 받은 횟수 합의 평균
    ## etc_sum : acc_id의 각 소속 혈맹에서 관찰 기간 동안 전투 활동을 한 캐릭터 수 합의 평균
    ## combat_play_time_sum : acc_id의 각 소속 혈맹에서 관찰 기간 동안 전투 활동을 한 캐릭터 수 합의 평균
    ## non_combat_play_time_sum : acc_id의 각 소속 혈맹에서 관찰 기간 동안 전투 활동을 한 캐릭터 수 합의 평균

    test1_pledge_28 = test1_pledge_arranged.drop(['char_id','acc_id'],1).drop_duplicates()
    test1_pledge_28_1 = test1_pledge_28.groupby('pledge_id').sum().rename(columns = {'play_char_cnt' : 'play_char_sum',
                                                                 'combat_char_cnt' : 'combat_char_sum',
                                                                 'pledge_combat_cnt' : 'pledge_combat_sum',
                                                                 'random_attacker_cnt' : 'random_attacker_sum',
                                                                 'random_defender_cnt' : 'random_defender_sum',
                                                                 'same_pledge_cnt' : 'same_pledge_sum',
                                                                 'temp_cnt' : 'temp_sum',
                                                                 'etc_cnt' : 'etc_sum',
                                                                 'combat_play_time' : 'combat_play_time_sum',
                                                                 'non_combat_play_time' : 'non_combat_play_time_sum'
                                                                })
    test1_pledge_28_2 = test1_pledge_28[['pledge_id','server']].drop_duplicates().groupby('pledge_id').count().rename(columns={'server':'play_server_cnt'})

    pledge_summary = test1_pledge_28_1.merge(test1_pledge_28_2,on='pledge_id')

    ## non_combat_pledge_cnt : acc_id의 각 소속 혈맹 중 비전투 혈맹의 수
    pledge_summary['is_non_combat_pledge'] = (pledge_summary['combat_play_time_sum']== 0) * 1

    ## conflict_pledge_cnt : acc_id의 각 소속 혈맹 중 동일혈맹 전투를 경함한 수
    pledge_summary['is_conflict_pledge'] = (pledge_summary['same_pledge_sum'] > 0) * 1

    ## combat_play_char_ratio : acc_id의 각 소속 혈맹 전투참여 캐릭터 비율의 평균
    pledge_summary['combat_play_char_ratio'] = pledge_summary['combat_char_sum']/pledge_summary['play_char_sum']

    pledge_summary=pledge_summary.drop(['day'],1)
    # numeric -> mean, binary -> sum
    df4 = test1_pledge_arranged[['acc_id','pledge_id']].drop_duplicates().merge(pledge_summary,how='left',on='pledge_id')
    mean = df4.groupby('acc_id').mean()[['play_char_sum','combat_char_sum','pledge_combat_sum','random_attacker_sum','random_defender_sum','same_pledge_sum','temp_sum','etc_sum','combat_play_time_sum','non_combat_play_time_sum','play_server_cnt','combat_play_char_ratio']]
    sm = df4.groupby('acc_id').sum()[['is_non_combat_pledge','is_conflict_pledge']]


    test1_pledge_arranged = train_pledge.sort_values(by=['day','acc_id','pledge_id','char_id','server'], axis=0)
    df1 = test1_pledge_arranged.iloc[:,0:5]
    df11= df1[['acc_id','char_id','pledge_id']].drop_duplicates()

    ## pledged_char_cnt : acc_id의 캐릭터 중 혈맹에 가입된 캐릭터의 수
    acc_1 = df11[['acc_id','char_id']].drop_duplicates().groupby("acc_id").count().rename(columns = {'char_id': 'pledged_char_cnt'})

    ## pledge_cnt : acc_id의 캐릭터들이 가입한 혈맹의 개수의 합
    acc_2 = df11[['acc_id','pledge_id']].drop_duplicates().groupby("acc_id").count().rename(columns = {'pledge_id': 'pledge_cnt'})

    ## same_pledged_char_cnt : acc_id의 캐릭터 중 동일혈맹에 가입한 캐릭터의 수
    df12 = df11.groupby(['acc_id','pledge_id']).count().rename(columns = {'char_id': 'n'})
    acc_3 = df12[df12['n']> 1].groupby("acc_id").sum().rename(columns = {'n':'same_pledged_char_cnt'})

    ## plural_pledge_cnt : acc_id의 캐릭터 중 동일혈맹에 있는 캐릭터가 2개 이상인 혈맹 수
    acc_4 = df12[df12['n']> 1].groupby('acc_id').count().rename(columns = {'n' : 'plural_pledge_cnt'})

    ## pledge_changed_char_cnt : acc_id의 혈맹 변경 캐릭터 수
    df2 =test1_pledge_arranged[['day','acc_id','char_id','pledge_id']].groupby(['acc_id','char_id','pledge_id']).count().groupby(['acc_id','char_id']).count().rename(columns = {'day' : 'num_pledge'})
    df2['num_pledge'] = df2['num_pledge']-1
    acc_5 =df2[df2['num_pledge']> 0].groupby('acc_id').count().rename(columns = {'num_pledge': 'pledge_changed_char_cnt'})

    ## pledge_changed_cnt : acc_id의 총 혈맹 변경 수
    acc_6 = df2.groupby('acc_id').sum().rename(columns = {'num_pledge': 'pledge_changed_cnt'})

    acc_table = acc_1.merge(acc_2,how='left',on =['acc_id'])
    acc_table = acc_table.merge(acc_3,how='left',on=['acc_id']).fillna(0).astype(int) #
    acc_table = acc_table.merge(acc_4,how='left',on=['acc_id']).fillna(0).astype(int)
    acc_table = acc_table.merge(acc_5,how='left',on=['acc_id']).fillna(0).astype(int)
    acc_table = acc_table.merge(acc_6,how='left',on=['acc_id']).fillna(0).astype(int)

    acc_table = acc_table.merge(mean,on='acc_id')
    df_temp_pled = acc_table.merge(sm,on='acc_id')

    df_pled = pd.merge(df_pled, df_temp_pled, on='acc_id', how='left').rename({'is_non_combat_pledge' : 'non_combat_pledge_cnt',
            'is_conflict_pledge' : 'conflict_pledge_cnt'})


    ### 6. merge
    df_out = pd.merge(df_acti, df_comb, on='acc_id', how='left')
    df_out = pd.merge(df_out, df_trad, on='acc_id', how='left')
    df_out = pd.merge(df_out, df_paym, on='acc_id', how='left')
    df_out = pd.merge(df_out, df_pled, on='acc_id', how='left')


    ### 7. etc
    merged_data = df_out

    ## level_group : 레벨 그룹 변수
    merged_data['level_group'] = np.where(merged_data['total_max_level']>10, 2, np.where(merged_data['total_max_level']<10, 0, 1))

    ## categorical 변수
    categorical_features = ['common_item_buy', 'common_item_sell', 'level_group', 'last_sell_item_type', 'last_buy_item_type']
    for cat_feature in categorical_features:
        tmp = pd.get_dummies(merged_data[cat_feature], prefix=cat_feature, drop_first=True)
        merged_data = pd.concat([merged_data, tmp], 1)
    merged_data = merged_data.drop(categorical_features, 1)
    merged_data = merged_data.fillna(0)

    user_feature = ['acc_id']
    label_feature = ['survival_time', 'amount_spent', 'survival_yn', 'amount_yn']
    ## 최종적으로 삭제할 columns
    remove_feature = ['chivalry_cnt',
     'dark_cnt', 'dragon_cnt', 'emperor_cnt', 'fairy_cnt', 'friend_pledge_cnt', 'friend_pledge_rate',
     'king_cnt', 'level_down_rate', 'level_keep_char_cnt', 'level_up_char_cnt', 'magic_cnt', 'mean_of_etc_cnt', 'mean_of_num_opponent',
     'mean_of_pledge_cnt', 'mean_of_random_attacker_cnt', 'mean_of_random_defender_cnt', 'mean_of_same_pledge_cnt',
     'mean_of_temp_cnt', 'sp_cls_cnt', 'sum_of_etc_cnt', 'sum_of_num_opponent', 'sum_of_pledge_cnt',  'sum_of_random_attacker_cnt',
     'sum_of_random_defender_cnt', 'sum_of_same_pledge_cnt', 'sum_of_temp_cnt', 'warrior_cnt', 'last7_pledge_combat_day', 'last7_pledge_combat_char']

    features_test = sorted(list(set(merged_data.columns) - set(remove_feature)))

    merged_data[features_test].to_csv(path_preprocess+"/"+file_name,index=None)


test1_day_range = (1,28)
test2_day_range = (1,28)
train_day_range = (1,28)

preproc('test1',test1_day_range)
preproc('test2',test2_day_range)
preproc('train',train_day_range)
