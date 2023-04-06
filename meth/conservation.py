import pyBigWig
import os
import numpy as np
#from .tools import convert_cpgs_notations

#bw_hg38_phylop = pyBigWig.open(os.path.join(os.path.dirname(__file__),"hg38.phyloP100way.bw"), "r")
#bw_hg38_phastcons = pyBigWig.open(os.path.join(os.path.dirname(__file__),"hg38.phastCons100way.bw"), "r")

#bw_hg19_phylop = pyBigWig.open(os.path.join(os.path.dirname(__file__),"hg19.phyloP100way.bw"), "r")
#bw_hg19_phastcons = pyBigWig.open(os.path.join(os.path.dirname(__file__),"hg19.phastCons100way.bw"), "r")

def convert_cpgs_notations(cpg):
    try:
        if '_' in cpg:
            ch, pos = cpg.split('_')
            pos_st = int(pos)
            pos_end = int(pos) + 1
            return '-'.join([':'.join([ch, str(pos_st)]), str(pos_end)])
        elif ':' in cpg:
            ch = cpg.split(':')[0]
            pos_st, pos_end = cpg.split(':')[1].split('-')
            pos_st, pos_end = int(pos_st), int(pos_end)
            return '_'.join([ch, str(pos_st)])
    except:
        return "None"

def get_score(cpg, bw, format_from='chr_pos', score='phyloP', assembly='hg38'):
    try:
        if format_from == 'chr_pos':
            cpg = convert_cpgs_notations(cpg)
        ch = cpg.split(':')[0]
        pos_st, pos_end = cpg.split(':')[1].split('-')
        pos_st, pos_end = int(pos_st), int(pos_end)
        return bw.values(ch, pos_st, pos_end)[0]
        # if score == 'phyloP':
        #     if assembly == 'hg38':
        #         return bw.values(ch, pos_st, pos_end)[0]
        #     elif assembly == 'hg19':
        #         return bw.values(ch, pos_st, pos_end)[0]
        #     elif assembly == 'mm10':
        #         return bw.values(ch, pos_st, pos_end)[0]
        # elif score == 'phastCons':
        #     if assembly == 'hg38':
        #         return bw.values(ch, pos_st, pos_end)[0]
        #     elif assembly == 'hg19':
        #         return bw.values(ch, pos_st, pos_end)[0]
        #     elif assembly == 'mm10':
        #         return bw.values(ch, pos_st, pos_end)[0]

    except:
        return np.nan