
### OLD TraceFearCondSession:

    def find_exp_boundaries(self):
        super().find_exp_boundaries()

        # Step 4. From there, interpolate where the various tone/shock periods are in the **Miniscope** timestamps
        # and then frame numbers. Expand all times to milliseconds (hence * 1000 in the below).

        # Get interpolated times.
        tone_onsets_rel = self.tone_onsets_def[self.periods] * 1000
        tone_offsets_rel = tone_onsets_rel[self.periods] + self.tone_duration*1000
        shock_onsets_rel = self.shock_onsets_def[self.periods] * 1000
        shock_offsets_rel = shock_onsets_rel[self.periods] + self.shock_duration*1000

        tone_onsets_adj = self.tone_onsets_def[self.periods] * 1000 + self.miniscope_exp_ts[self.start_idx] 
        tone_offsets_adj = tone_onsets_adj[self.periods] + self.tone_duration*1000
        shock_onsets_adj = self.shock_onsets_def[self.periods] * 1000 + self.miniscope_exp_ts[self.start_idx]
        shock_offsets_adj = shock_onsets_adj[self.periods] + self.shock_duration*1000

        # to correct for detecting tone_onsets using tone_onsets_rel
        ts_exp_start = self.miniscope_exp_ts[self.start_idx]
        ts_exp_start = 0

        # Find closest corresponding Miniscope timestamps and get the frame numbers.
        for ts in tone_onsets_rel:
            found_ts = np.where(abs(self.tstamp_miniscope - (ts + ts_exp_start)) < self.tstamp_tol)
            self.tone_onsets.append(found_ts[0][0])
        for ts in tone_offsets_rel:
            found_ts = np.where(abs(self.tstamp_miniscope - (ts + ts_exp_start)) < self.tstamp_tol)
            self.tone_offsets.append(found_ts[0][0])
        for ts in shock_onsets_rel:
            found_ts = np.where(abs(self.tstamp_miniscope - (ts + ts_exp_start)) < self.tstamp_tol)
            self.shock_onsets.append(found_ts[0][0])
        for ts in shock_offsets_rel:
            found_ts = np.where(abs(self.tstamp_miniscope - (ts + ts_exp_start)) < self.tstamp_tol)
            self.shock_offsets.append(found_ts[0][0])

        for ts in tone_onsets_adj:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_onsets_adj.append(found_ts[0][0])
        for ts in tone_offsets_adj:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_offsets_adj.append(found_ts[0][0])
        for ts in shock_onsets_adj:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.shock_onsets_adj.append(found_ts[0][0])
        for ts in shock_offsets_adj:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.shock_offsets_adj.append(found_ts[0][0])

        # Now find post-shock, based on the already found tone/shock boundaries
        for i in range(len(self.tone_onsets)):
            if i == len(self.tone_onsets)-1:
                self.post_shock_onsets.append(self.shock_offsets[i])
                self.post_shock_offsets.append(self.miniscope_exp_fnum[self.stop_idx])
            else:
                self.post_shock_onsets.append(self.shock_offsets[i])
                self.post_shock_offsets.append(self.tone_onsets[i+1])

        for i in range(len(self.tone_onsets_adj)):
            if i == len(self.tone_onsets_adj)-1:
                self.post_shock_onsets_adj.append(self.shock_offsets_adj[i])
                self.post_shock_offsets_adj.append(self.miniscope_exp_fnum[self.stop_idx])
            else:
                self.post_shock_onsets_adj.append(self.shock_offsets_adj[i])
                self.post_shock_offsets_adj.append(self.tone_onsets_adj[i+1])


### OLD TestBSession
                
                
    def find_exp_boundaries(self):
        super().find_exp_boundaries()

        # Step 4. From there, interpolate where the various tone/shock periods are in the **Miniscope** timestamps
        # and then frame numbers. Expand all times to milliseconds (hence * 1000 in the below).

        # Get interpolated times.
        tone_onsets_rel = self.tone_onsets_def[self.periods] * 1000
        tone_offsets_rel = tone_onsets_rel[self.periods] + self.tone_duration*1000

        tone_onsets_adj = self.tone_onsets_def[self.periods] * 1000 + self.miniscope_exp_ts[self.start_idx] 
        tone_offsets_adj = tone_onsets_adj[self.periods] + self.tone_duration*1000

        # to correct for detecting tone_onsets using tone_onsets_rel
        #ts_exp_start = self.miniscope_exp_ts[self.start_idx]
        ts_exp_start = 0

        # Find closest corresponding Miniscope timestamps and get the frame numbers.
        for ts in tone_onsets_rel:
            found_ts = np.where(abs(self.tstamp_miniscope (ts + ts_exp_start)) < self.tstamp_tol)
            self.tone_onsets.append(found_ts[0][0])
        for ts in tone_offsets_rel:
            found_ts = np.where(abs(self.tstamp_miniscope (ts + ts_exp_start)) < self.tstamp_tol)
            self.tone_offsets.append(found_ts[0][0])

        for ts in tone_onsets_adj:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_onsets_adj.append(found_ts[0][0])
        for ts in tone_offsets_adj:
            found_ts = np.where(abs(self.tstamp_miniscope - ts) < self.tstamp_tol)
            self.tone_offsets_adj.append(found_ts[0][0])
    
        # Now find post-tone, based on the already found tone/shock boundaries
        for i in range(len(self.tone_onsets)):
            self.post_tone_onsets.append(self.tone_offsets[i])
            self.tone_post_tone_onsets.append(self.tone_onsets[i])

            if i == len(self.tone_onsets)-1:
                self.post_tone_offsets.append(self.miniscope_exp_fnum[self.stop_idx])
                self.tone_post_tone_offsets.append(self.miniscope_exp_fnum[self.stop_idx])
            else:
                self.post_tone_offsets.append(self.tone_onsets[i+1])
                self.tone_post_tone_offsets.append(self.tone_onsets[i+1])



### OLD BehaviourSession
                
    def find_exp_boundaries(self):

        [self.fnum_miniscope, self.tstamp_miniscope, self.fnum_behavcam, self.tstamp_behavcam] \
            = self.get_timestamps()

        # Logic: 
        # Step 1. Find frames in actual videofiles where the session starts and ends to mark the boundaries
        #     of the experiment. These go in light_frames (required; provided in constructor).

        # Step 2. Find the corresponding timestamps of these frames in the **BehavCam** timestamps file.
        self.behavcam_exp_ts = [self.tstamp_behavcam.iloc[self.session_bounds[self.start_idx]], \
            self.tstamp_behavcam.iloc[self.session_bounds[self.stop_idx]]]

        # Step 3. Find the closest corresponding timestamps of these frames in the **Miniscope** timestamps file.
        # These then become directly the frame numbers.
        beg = np.where(abs(self.tstamp_miniscope - self.behavcam_exp_ts[self.start_idx]) < self.tstamp_tol)
        end = np.where(abs(self.tstamp_miniscope - self.behavcam_exp_ts[self.stop_idx]) < self.tstamp_tol)
        # there may be more than one in beg, end, so we just take the first, which is also the closest
        self.miniscope_exp_fnum = [beg[0][0], end[0][0]] 
        self.miniscope_exp_ts = [self.tstamp_miniscope[self.miniscope_exp_fnum[self.start_idx]], \
            self.tstamp_miniscope[self.miniscope_exp_fnum[self.stop_idx]]]