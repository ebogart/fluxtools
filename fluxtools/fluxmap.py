def fluxmap(fluxes, sdict, threshold=1e-7):
    species_total_in = {}
    species_total_out = {}
    species_input_breakdown = {}
    species_output_breakdown = {}

    for r,f in fluxes.items():
        if r not in sdict:
            print 'Skipping ' + r
            continue
        if r.startswith('reverse_'):
            r = r[8:]
            f = -1.0 * f
        st = sdict[r]
        for s,c in st.items():
            if s not in species_total_in:
                species_total_in[s] = 0
                species_total_out[s] = 0
                species_input_breakdown[s] = {}
                species_output_breakdown[s] = {} 
            if c*f > threshold:
                species_total_in[s] += c*f
                species_input_breakdown[s][r] = c*f
            if c*f < -1.0*threshold:
                species_total_out[s] += -1.0*c*f
                species_output_breakdown[s][r] = -1.0*c*f

    return species_total_in, species_total_out, species_input_breakdown, species_output_breakdown


