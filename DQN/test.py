

def eps_anneal(inital,final,total_frames):
    return (inital - final) / total_frames

print(eps_anneal(1,.1,100000))
