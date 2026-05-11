    # PSTH_2 no averaging

    frames_lookaround = 40
    frames_save = 200
    num_shuffles=100
    percentile=95.0

    crossreg_mice=TFC_cond_crossreg
    #crossreg_mice=TFC_B_B_1wk_crossreg

    mapping_type='full'
    #mapping_type=mapping_TFC_cond
    #mapping_type=mapping_TFC_cond_Test_B
    #mapping_type=mapping_TFC_cond_Test_B_Test_B_1wk
    #mapping_type=mapping_LT2_TFC_cond

    session=TFC_cond
    stim='shock'

    #snippet_len = frames_lookaround + frames_save # frames_lookaround * 2
    snippet_len = frames_lookaround * 2
    save_post = frames_lookaround + frames_save
    save_len = frames_lookaround * 2 + frames_save
    group_PSTH = dict()  # can be either spike rates or average spike intensities, depending on use_peakval
    #group_PSTH['hM3D'] = np.array([])
    #group_PSTH['hM4D'] = np.array([])
    #group_PSTH['mCherry'] = np.array([])

    tot_cells = dict()
    tot_cells['hM3D'] = dict()
    tot_cells['hM4D'] = dict()
    tot_cells['mCherry'] = dict()

    sig_cells = dict()
    sig_cells['hM3D'] = dict()
    sig_cells['hM4D'] = dict()
    sig_cells['mCherry'] = dict()

    frac_tots = dict()
    frac_tots['hM3D'] = dict()
    frac_tots['hM4D'] = dict()
    frac_tots['mCherry'] = dict()

    rng = default_rng()

    print('*** MAPPING: {}'.format(mapping_type))
    for group, mice in mice_per_group.items():

        print('\nin {} {}'.format(group, mice))
        for m in range(len(mice)):
            mouse = mice[m]
            #if mouse in ['G07', 'G14', 'G15', 'G20', 'G12']:
            #    continue
            print('{} {}'.format(mouse, group))
            s = session[mouse]
            crossreg = crossreg_mice[mouse]
            frac_tots[group][mouse] = []
            sig_cells[group][mouse] = []
            tot_cells[group][mouse] = []

            if stim == 'tone':
                onsets = s.tone_onsets
                offsets = s.tone_offsets
            if stim == 'shock':
                onsets = s.shock_onsets
                offsets = s.shock_offsets

            if mapping_type == 'full':
                #C = s.C_zarr['C']
                #C = s.C
                C = s.S
                #C = s.S_immobility
            else:
                indeces = []
                df_mapping = crossreg.get_mappings_cells(mapping_type=mapping_type)
                s_df_col = s.get_df_col()
                for i in range(len(df_mapping)):
                    cell_s = int(float(df_mapping[s_df_col].iloc[i]))
                    try:
                        #idx = np.where(s.C_zarr['unit_id']==cell_s)[0][0]
                        idx = np.where(s.C_idx==cell_s)[0][0]
                        indeces.append(idx)
                    except IndexError as error:
                        # sometimes the minian-saved zarr files don't contain the cells from the crossreg mapping;
                        # so just skip this row
                        print('***WARNING: could not find {} ({})'.format(cell_s, s.session_type))
                        continue
                C = s.S[indeces,:]
            tot_cells[group][mouse].append(C.shape[0])

            C_responses = np.zeros((C.shape[0], frames_lookaround*2))
            C_responses_save = np.zeros((C.shape[0], save_len))
            sig_cells_total_mouse = np.ndarray(0)

            #onsets_wanted = [onsets[0]]
            #onsets_wanted = [onsets[-1]]
            onsets_wanted = onsets
            for on,on_num in zip(onsets_wanted,range(len(onsets_wanted))):
                print('on {}'.format(on), end='')
                #on = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround))
                period = range(on - frames_lookaround, on + frames_lookaround)
                period_save = range(on - frames_lookaround, on + save_post)
                #C_responses = C_responses + C[:,period]
                C_responses = C_responses + C[:,period]
                C_responses_save = C_responses_save + C[:,period_save]

                post = np.mean(C_responses[:, range(frames_lookaround, len(period))],1)
                pre = np.mean(C_responses[:, range(0, frames_lookaround)],1)
                '''
                post_save = np.mean(C_responses_save[:, range(frames_lookaround, len(period))],1)
                pre_save = np.mean(C_responses_save[:, range(0, frames_lookaround)],1)            
                C_responses_save = post_save - pre_save
                C_responses = post - pre
                '''
                #C_binary = (post - pre) / (post + pre)
                C_binary = post - pre

                #C_binary = np.mean(C[:, range(on, on + frames_lookaround)],1) - \
                #    np.mean(C[:, range(on - frames_lookaround, on)],1)

                # Allocate matrix where we have the response-values for all of the shuffles (columns) for each cell (rows).
                C_shuffles = np.zeros((C.shape[0], len(period), num_shuffles))
                C_shuffles_binary = np.zeros((C.shape[0], num_shuffles))
                #C_period_values = np.zeros((C.shape[0], len(period) * 2, num_shuffles)) # for entire onset,offset period
                for i in range(num_shuffles):
                    print('.'.format(on),end='')                
                    shuffle_times = rng.choice(np.arange(frames_lookaround, C.shape[1] - frames_lookaround), num_shuffles)
                    C_rolled = np.roll(C, shuffle_times, axis=0)

                    '''
                    C_rolled_responses = np.zeros((C_rolled.shape[0], frames_lookaround*2))
                    for on in onsets:
                        period = range(on - frames_lookaround, on + frames_lookaround)
                        C_rolled_responses = np.add(C_rolled_responses, C_rolled[:,period])
                    C_rolled_responses /= len(onsets)
                    '''

                    C_rolled_responses = C_rolled[:, range(shuffle_times[i] - frames_lookaround, shuffle_times[i] + frames_lookaround)]
                    post_shuffle = np.mean(C_rolled_responses[:, range(frames_lookaround, len(period))],1)
                    pre_shuffle = np.mean(C_rolled_responses[:, range(0, frames_lookaround)],1)
                    #C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                    C_shuffles_binary[:,i] = post_shuffle - pre_shuffle

                    ##
                    '''
                    post_shuffle = np.mean(C_rolled[:, range(shuffle_times[i], shuffle_times[i] + frames_lookaround)])
                    pre_shuffle = np.mean(C_rolled[:, range(shuffle_times[i] - frames_lookaround, shuffle_times[i])])
                    #C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                    C_shuffles_binary[:,i] = post_shuffle - pre_shuffle
                    #C_shuffles_binary[:,i] = np.mean(C_rolled[:,range(shuffle_times[i]-int(len(period)/2))])
                    '''
                    ##
                    '''
                    C_shuffles[:,:,i] = C_rolled[:,period]
                    half_period = int(len(period)/2)
                    post_shuffle = np.mean(C_shuffles[:, range(half_period, len(period)), i],1)
                    pre_shuffle = np.mean(C_shuffles[:, range(0, half_period), i],1)
                    C_shuffles_binary[:,i] = (post_shuffle - pre_shuffle) / (post_shuffle + pre_shuffle)
                    '''
                    #np.nan_to_num(C_shuffles_binary,copy=False)
                    ##C_shuffles_binary[:,i] = post_shuffle - pre_shuffle
                #C_percentile = np.percentile(C_shuffles, percentile, axis=(1,2)) # Will compute percentiles along 0th axis, i.e., cells
                C_percentile = np.percentile(C_shuffles_binary, percentile, axis=1)
                sig_cells_mouse = np.where((C_binary > C_percentile)==True)[0]
                sig_cells_total_mouse = np.append(sig_cells_total_mouse, sig_cells_mouse)
                frac_tots[group][mouse].append(len(sig_cells_mouse) / C.shape[0])
                #frac_tots[group].append(len(sig_cells_mouse))
                sig_cells[group][mouse].append(sig_cells_mouse)

                if group not in group_PSTH.keys():
                    group_PSTH[group] = C_responses_save[sig_cells_mouse,:]
                else:
                    group_PSTH[group] = np.vstack((group_PSTH[group], C_responses_save[sig_cells_mouse,:]))
                print('')

        #frac_tots[group].append(len(np.unique(sig_cells_total_mouse)) / C.shape[0])
    plt.figure()
    plt.plot(np.mean(group_PSTH['hM3D'],0))
    plt.title('hM3D')
    plt.figure()
    plt.plot(np.mean(group_PSTH['hM4D'],0))
    plt.title('hM4D')
    plt.figure()
    plt.plot(np.mean(group_PSTH['mCherry'],0))
    plt.title('mCherry')

    print('tot cells hM3D ', [x for x in tot_cells['hM3D'].values()])
    print('tot cells hM4D ', [x for x in tot_cells['hM4D'].values()])
    print('tot cells mCherry ', [x for x in tot_cells['mCherry'].values()])
    print('frac tots hM3D ', [x for x in frac_tots['hM3D'].values()])
    print('frac tots hM4D ', [x for x in frac_tots['hM4D'].values()])
    print('frac tots mCherry ', [x for x in frac_tots['mCherry'].values()])
    print('sig cells hM3D ', [len(x) for x in sig_cells['hM3D'].values()])
    print('sig cells hM4D ', [len(x) for x in sig_cells['hM4D'].values()])
    print('sig_cells mCherry ', [len(x) for x in sig_cells['mCherry'].values()])
