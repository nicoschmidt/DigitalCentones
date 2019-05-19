'''
Created on Nov 14, 2018

@author: Nico Schmidt
'''
import codecs
import numpy as np

import heapq
import xml.etree.ElementTree as ET
import re
import unicodedata
from smith_waterman import SmithWaterman
import pickle


######################################
## THE XML FILES WE WANT TO ANALIZE ##
######################################
chrpat_xml = 'tlg2022.tlg003.opp-grc1.xml'
medea_txt = 'medea.txt' # The Medea text we only have as plain text, not as T’EI-XML !




#####################################
## SOME TEXT PREPARATION FUNCTIONS ##
#####################################
def remove_punctuation(text):
    '''
    Remove all sorts of punctuation characters.
    Not exhaustive, just everything I bumped into so far, feel free to extend.
    '''
    return text.replace('!', '')\
               .replace("'", '')\
               .replace('(', '')\
               .replace(')', '')\
               .replace(',', '')\
               .replace('.', '')\
               .replace(':', '')\
               .replace(';', '')\
               .replace('[', '')\
               .replace(']', '')\
               .replace('·', '')\
               .replace('ʼ', '')\
               .replace('̓', '')\
               .replace('·', '')\
               .replace('᾿', '')\
               .replace('—', '')\
               .replace('‘', '')\
               .replace('’', '')\
               .replace('“', '')\
               .replace('”', '')\
               .replace('„', '')\
               .replace('†', '')

def replace_nongreek(text):
    '''
    Replace non-Greek characters by Greek ones,
    e.g. latin E (capital e) by Greek Ε (epsilon)
    '''
    return text.replace('E', 'Ε')

def prepare_text(text):
    '''
    Apply all of the above and make everything lower case.
    '''
    return replace_nongreek(remove_punctuation(text)).lower()

def strip_accents(s):
    '''
    remove all accents/diacritics as far as defined in Unicode data (unfortunately does not remove everything)
    found here: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string
    Iota is a special case: it is used as diacritics if following eta or omega.
    '''
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')\
           .replace('ηι', 'η')\
           .replace('ωι', 'ω')




############################################
## OUR DATA STRUCTURES TO STORE THE TEXTS ##
############################################
class Line(object):
    '''
    A single line of a work
    '''
    def __init__(self,
                work_id='',
                idx=-1,
                no=-1,
                speaker='',
                text_raw='',
                text='',
                text_no_diacritics='',
                text_lemma='',
                lemma_ids=[]):
        self.work_id = work_id # which work, e.g. TLG identifier
        self.idx = idx # line index
        self.no = no # line number
        self.speaker = speaker # current speaker (if applicable)
        self.text_raw = text_raw # text without modifications
        self.text = text # text with punctuation etc. removed
        self.text_no_diacritics = text_no_diacritics # text with diacritics, punctuation etc. removed
        self.text_lemma = text_lemma # text with words replaced by their lemma form
        self.lemma_ids = lemma_ids # ids of lemmata
    
    def __repr__(self):
        return 'work_id:            {}\n'.format(self.work_id) + \
               'idx:                {}\n'.format(self.idx) + \
               'no:                 {}\n'.format(self.no) + \
               'speaker:            {}\n'.format(self.speaker) + \
               'text_raw:           {}\n'.format(self.text_raw) + \
               'text:               {}\n'.format(self.text) + \
               'text_no_diacritics: {}\n'.format(self.text_no_diacritics) + \
               'text_lemma:         {}\n'.format(self.text_lemma) + \
               'lemma_ids:          {}'.format(self.lemma_ids)


class Work(list):
    '''
    A work (basically a list of lines).
    '''
    def __init__(self, work_id='', title='', author='', *args):
        list.__init__(self, *args)
        self.work_id = work_id
        self.title = title
        self.author = author

class Alignment(object):
    '''
    Alignment info of two lines. Contains the following information:
    score:                  the score of this alignment
    source_text_id:         the ID of the source text
    source_line_idx:        the index of the line in the source text
    target_line_idx:        the index of the line in the target text
    alignment:              the alignment info as list of indexes (i,j) defining which symbols of the lines are aligned
                            i is the index of the symbol in the source line, j is the index of the symbol in the target line
    aligned_target_sequece: the aligned target sequence
    aligned_source_sequece: the aligned source sequence
    '''
    def __init__(self, score, source_text_id, source_line_idx, target_line_idx, alignment, aligned_target_sequence, aligned_source_sequence):
        self.score = score
        self.source_text_id = source_text_id
        self.source_line_idx = source_line_idx
        self.target_line_idx = target_line_idx
        self.alignment = alignment
        self.aligned_target_sequence = aligned_target_sequence
        self.aligned_source_sequence = aligned_source_sequence
    
    # for heapsort to work with this class we need to implement ordering functions:
    def __eq__(self, other):
        return (self.score == other.score)
    def __ne__(self, other):
        return (self.score != other.score)
    def __lt__(self, other):
        return (self.score < other.score)
    def __le__(self, other):
        return (self.score <= other.score)
    def __gt__(self, other):
        return (self.score > other.score)
    def __ge__(self, other):
        return (self.score >= other.score)
    
    def __repr__(self):
        return str((self.score, [(i,j) for i,j in self.alignment]))




####################
## LOAD XML FILES ##
####################
def load_xml(fname):
    '''
    Read texts from xml files that are T’EI-XML formatted.
    '''
    ns_map = dict([(v,k) for k,v in ET._namespace_map.items()])
    with open(fname, 'r') as f: # load manually and remove namespace
        xmlstring = f.read()
        xmlstring = re.sub(r"""\s(xmlns="[^"]+"|xmlns='[^']+')""", '', xmlstring, count=1) # remove annoying namespace
    root = ET.fromstring(xmlstring)
    parent_map = {c:p for p in root.iter() for c in p}
    for div_top in root.findall('text/body/div'):
        if div_top.attrib['type']=='edition' and div_top.attrib['{'+ns_map['xml']+'}lang']=='grc':
            _id = div_top.get('n')
    lines = Work(_id)
    i=0
    for line in root.iter('l'):
        speaker = parent_map[line].find('speaker')
        if speaker is None:
            speaker = ''
        else:
            speaker = speaker.text
        n = line.get('n')
        n = int(n) if n else -1 # make n integer
        text = prepare_text(line.text)
        text_no_diacritics = strip_accents(text)
        lines.append(Line(work_id=_id,
                          idx=i,
                          no=n,
                          speaker=speaker,
                          text_raw=line.text,
                          text=text,
                          text_no_diacritics=text_no_diacritics))
        i+=1
    return lines

def load_plain_text(fname, text_id, title='', author=''):
    '''
    Read texts from plain text file.
    '''
    with codecs.open(fname, encoding='utf-8') as fid:
        text_file = fid.read().replace(u'\xa0', u' ')
    lines = []
    for i,text_raw in enumerate(text_file.split(u'\n')):
        text = prepare_text(text_raw)
        text_no_diacritics = strip_accents(text)
        lines.append(Line(work_id=text_id,
                          idx=i,
                          no=i+1,
                          text_raw=text_raw,
                          text=text,
                          text_no_diacritics=text_no_diacritics))
    lines = Work(text_id, title, author, lines)
    return lines

lines_chr_pat = load_xml(chrpat_xml)
for i in range(30):
    lines_chr_pat[i].no = -30+i # hack to re-number preface lines
for i in range(30, len(lines_chr_pat)):
    lines_chr_pat[i].no = -29+i # hack to re-number lines
lines_medea = load_plain_text(medea_txt, 'Medea', 'Medea', 'Euripides')


# print all characters in the corpus:
alphabet = np.unique([c for lines in [lines_chr_pat, lines_medea] for line in lines for c in line.text])
for a in alphabet:
    print('{} "{}" {}'.format(hex(ord(a)), a, unicodedata.name(a, 'not defined')))
alphabet_no_diacritics = np.unique([c for lines in [lines_chr_pat, lines_medea] for line in lines for c in line.text_no_diacritics])
for a in alphabet_no_diacritics:
    print('{} "{}" {}'.format(hex(ord(a)), a, unicodedata.name(a, 'not defined')))

# # find latin letter a
# for lines in [lines_chr_pat, lines_medea]:
#     for line in lines:
#         if "a" in line.text_no_diacritics:
#             print('a found in {}, line {}: "{}" ("{}")'.format(lines.work_id, line.idx, line.text_raw, line.text))
# 
# for k in morph_dict:
#     if isinstance(k, str):
#         if 'νακτος' in k:
#             print(k)


def load_line_to_line_map(fname_line_to_line_map):
    '''
    load mapping of target lines to source work and source line.
    Input file needs to be a text file with one header row and remaining rows structured as
    '<source_work_number>\t<source_work_line_number>\t<target_line_number>...\n'
    output map is a dictionary with target_line_number as keys and tuples (<source_work_id>, <source_work_number>) as values
    '''
    line_to_line_map = np.loadtxt(fname_line_to_line_map, int, delimiter='\t', skiprows=1, usecols=[2,0,1])
    source_work_names = ['Agamemnon',
                         'Prometheus',
                         'Alkestis',
                         'Andromache',
                         'Bakchen',
                         'Hekabe',
                         'Helena',
                         'Hippolytos',
                         'Iphigenie in Aulis',
                         'Iphigenie in Tauris',
                         'Medea',
                         'Orestes',
                         'Ph\"onikerinnen',
                         'Rhesos',
                         'Troerinnen',
                         'Ilias',
                         'Alexandra']
    return dict([(target_line_no,(source_work_names[source_work_id-1], source_line_id)) for target_line_no,source_work_id, source_line_id in line_to_line_map])

fname_line_to_line_map = 'Christos.paschon.1.2a.csv'
line_to_line_map = load_line_to_line_map(fname_line_to_line_map)



def find_line_to_line_alignments(target_lines, source_lines_dict, alignment_function, n_max_alignments, line_form_func=lambda line:line.text, line_to_line_map=None):
    '''
    Do pairwise alignment of all lines of the target text with all lines of the source texts and find best alignments
    '''
    n_target_lines = len(target_lines)
    alignments = []
    print('searching all verses for alignments...')
    
    # go through all target lines
    for i,target_line in enumerate(target_lines):
        alignments_i = []
        
        # print status
        if i%10==0:
            print('{0}/{1} ({2:3.0f}%)'.format(i,n_target_lines, i/float(n_target_lines)*100))
        
        if line_to_line_map is None:
            # go through all sources texts
            for source_id, source_lines in source_lines_dict.items():
                
                # go through all lines of source text
                for source_line in source_lines:
                    
                    # do line-to-line alignment
                    alignments_ij = alignment_function(line_form_func(source_line), line_form_func(target_line))
                    
                    # store best alignment result if found
                    if alignments_ij:
                        (score, alignment, aligned_source_sequece, aligned_target_sequece) = alignments_ij[0] # get first (best) alignment
                        a = Alignment(score, source_id, source_line.idx, target_line.idx, alignment, aligned_target_sequece, aligned_source_sequece)
                        
                        # store it as tuple (score, alignment_object)
                        heapq.heappush(alignments_i, a)
                        
            # store the best n_max_alignments in alignment list
            # note: list is sorted in ascending order, so we need the last n_max_alignments elements to get the highest scores
            alignments_i = [heapq.heappop(alignments_i) for _ in range(len(alignments_i))]
            alignments.append(alignments_i[-n_max_alignments:][::-1])
            
        else:
            # go through line_to_line_map
            if target_line.no in line_to_line_map:
                (source_id, source_line_no) = line_to_line_map[target_line.no]
                if source_id in source_lines_dict:
                    source_lines = source_lines_dict[source_id]
                    source_line = [source_line for source_line in source_lines if source_line.no==source_line_no]
                    if source_line:
                        source_line = source_line[0]
                        alignments_ij = alignment_function(line_form_func(source_line), line_form_func(target_line))
                        if alignments_ij:
                            (score, alignment, aligned_source_sequece, aligned_target_sequece) = alignments_ij[0] # get first (best) alignment
                            alignments_i.append(Alignment(score, source_id, source_line.idx, target_line.idx, alignment, aligned_target_sequece, aligned_source_sequece))
            alignments.append(alignments_i)
    return alignments


def save_alignments(alignments, alignments_outfile):
    pickle.dump(alignments, open(alignments_outfile, 'wb'))

def load_alignment(alignments_outfile):
    return pickle.load(open(alignments_outfile, 'rb'))

def save_alignments_csv(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func=lambda line:line.text, min_score=0):
    '''
    write aignment info and strings to csv file.
    '''
    
    # find max number of alignments
    n_max_alignments = max([len(a) for a in alignments])
    
    # save found alignments to csv file
    with codecs.open(alignments_outfile+'.csv', 'w', encoding='utf-8') as fid:
        
        # write header
        fid.write(u'Target Line ID;Target Line No.;Speaker;Target Line Text')
        for _ in range(n_max_alignments):
            fid.write(u';Source Work')
            fid.write(u';Source Line ID')
            fid.write(u';Source Line Text')
            fid.write(u';Alignment Text')
            fid.write(u';Alignment Score')
            fid.write(u';Alignment Target Start idx')
            fid.write(u';Alignment Target End idx')
            fid.write(u';Alignment Source Start idx')
            fid.write(u';Alignment Source End idx')
        fid.write(u'\n')
        
        # go through all target lines
        for target_line, alignments_i in zip(target_lines,alignments):
            
            # write target line info
            fid.write(u'{0};{1};"{2}";"{3}"'.format(target_line.idx, target_line.no, target_line.speaker, line_form_func(target_line)))
            
            # go through all alignments
            for j in range(n_max_alignments):
                # write source line info and alignment results
                if j<len(alignments_i) and alignments_i[j].score>=min_score:
                    alignment = alignments_i[j]
                    text = line_form_func(source_lines_dict[alignment.source_text_id][alignment.source_line_idx])
                    fid.write(u';"{}"'.format(alignment.source_text_id))
                    fid.write(u';{}'.format(alignment.source_line_idx))
                    fid.write(u';"{}"'.format(text))
                    fid.write(u';"{}"'.format(alignment.aligned_target_sequence))
                    fid.write(u';{0:.0f}'.format(alignment.score))
                    fid.write(u';{}'.format(alignment.alignment[0][1]))
                    fid.write(u';{}'.format(alignment.alignment[-1][1]))
                    fid.write(u';{}'.format(alignment.alignment[0][0]))
                    fid.write(u';{}'.format(alignment.alignment[-1][0]))
                else:
                    fid.write(u';;;;;;;;;')        
            fid.write(u'\n')


def save_alignments_html(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func=lambda line:line.text, min_score=0):
    '''
    write aignment info and strings to html file.
    '''
    
    # save found alignments to html file
    with codecs.open(alignments_outfile+'.html', 'w', encoding='utf-8') as fid:
        
        # string replace function for handling lists
        make_string = lambda string,space_before,space_after: (' ' if space_before else '')+' '.join(string)+(' ' if space_after else '') if isinstance(string, (list,tuple)) else string
        
        # write header
        fid.write(u'<!DOCTYPE html>\n')
        fid.write(u'<html>\n')
        fid.write(u'<head>\n')
        fid.write(u'<style>\n')
        fid.write(u'th {\n')
        fid.write(u'\ttext-align: center;\n')
        fid.write(u'}\n')
        fid.write(u'td {\n')
        fid.write(u'\ttext-align: center;\n')
        fid.write(u'}\n')
        fid.write(u'.match {\n')
        fid.write(u'\tbackground-color:Yellow\n')
        fid.write(u'}\n')
        fid.write(u'</style>\n')
        fid.write(u'</head>\n')
        fid.write(u'<body>\n')
        fid.write(u'<table>\n')
        fid.write(u'<tr>\n')
        fid.write(u'\t<th>Target Line ID</th>\n')
        fid.write(u'\t<th>Target Line No.</th>\n')
        fid.write(u'\t<th>Target Line Text</th>\n')
        fid.write(u'\t<th>Source Line Text</th>\n')
        fid.write(u'\t<th>Source Line No.</th>\n')
        fid.write(u'\t<th>Source Text ID</th>\n')
        fid.write(u'\t<th>Alignment Score</th>\n')
        fid.write(u'</tr>\n')
        
        # go through all target lines
        for target_line, alignments_i in zip(target_lines,alignments):
            
            # write target line info
#             fid.write(u'<tr><td>{0}</td><td>{1}</td><td>{2}</td>'.format(target_line.idx, target_line.no, line_form_func(target_line)))
            
            if alignments_i and alignments_i[0].score>=min_score:
                # write source line info and alignment results
                alignment = alignments_i[0]
                target_text = line_form_func(target_line)
                source_text = line_form_func(source_lines_dict[alignment.source_text_id][alignment.source_line_idx])
                no = source_lines_dict[alignment.source_text_id][alignment.source_line_idx].no
                
                fid.write(u'<tr><td>{0}</td><td>{1}</td>'.format(target_line.idx, target_line.no))
                fid.write(u'<td>{}<span class="match">{}</span>{}</td>'.format(make_string(target_text[:alignment.alignment[0][1]], 0, 1),
                                                                               make_string(target_text[alignment.alignment[0][1]:alignment.alignment[-1][1]+1], 0, 0),
                                                                               make_string(target_text[alignment.alignment[-1][1]+1:], 1, 0)))
                fid.write(u'<td>{}<span class="match">{}</span>{}</td>'.format(make_string(source_text[:alignment.alignment[0][0]], 0, 1),
                                                                               make_string(source_text[alignment.alignment[0][0]:alignment.alignment[-1][0]+1], 0, 0),
                                                                               make_string(source_text[alignment.alignment[-1][0]+1:], 1, 0)))
#                 fid.write(u'<td>{}</td>'.format(source_text))
                fid.write(u'<td>{}</td>'.format(no))
                fid.write(u'<td>{}</td>'.format(alignment.source_text_id))
                fid.write(u'<td>{0:.0f}</td>'.format(alignment.score))
            else:
                fid.write(u'<tr><td>{0}</td><td>{1}</td><td>{2}</td>'.format(target_line.idx, target_line.no, make_string(line_form_func(target_line), 0, 0)))
                fid.write(u'<td></td><td></td><td></td><td></td>')
            fid.write(u'</tr>\n')
        fid.write(u'</table>\n</body>\n</html>\n')
    


#####################################################################
## Character-based line-to-line alignments on text with diacritics ##
#####################################################################
string_mapping_function = lambda s:s # identity
smith_waterman = SmithWaterman(match_score=5,
                               mismatch_score=-1,
                               gap_score=-1,
                               n_max_alignments=1,
                               min_score_treshold=0,
                               string_mapping_function=string_mapping_function)
line_form_func = lambda line:line.text
alignments_outfile = 'alignments_character-based_with_diacritics'
target_lines = lines_chr_pat
source_lines_dict = {lines_medea[0].work_id:lines_medea}
alignments = find_line_to_line_alignments(target_lines, source_lines_dict, smith_waterman.align, 3, line_form_func)
save_alignments(alignments, alignments_outfile)
save_alignments_csv(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)
save_alignments_html(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)


########################################################################
## Character-based line-to-line alignments on text without diacritics ##
########################################################################
smith_waterman = SmithWaterman(match_score=4,
                               mismatch_score=-1,
                               gap_score=-1,
                               n_max_alignments=1,
                               min_score_treshold=0)
line_form_func = lambda line:line.text_no_diacritics
alignments_outfile = 'alignments_character-based_without_diacritics'
target_lines = lines_chr_pat
source_lines_dict = {lines_medea[0].work_id:lines_medea}
alignments = find_line_to_line_alignments(target_lines, source_lines_dict, smith_waterman.align, 3, line_form_func)
save_alignments(alignments, alignments_outfile)
save_alignments_csv(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)
save_alignments_html(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func, 4)



###################################################################
## word-based line-to-line alignments on text without diacritics ##
###################################################################
smith_waterman = SmithWaterman(match_score=2,
                               mismatch_score=-1,
                               gap_score=-1,
                               n_max_alignments=1,
                               min_score_treshold=0)
line_form_func = lambda line:line.text_no_diacritics.split(' ')
alignments_outfile = 'alignments_word-based_without_diacritics'
target_lines = lines_chr_pat
source_lines_dict = {lines_medea[0].work_id:lines_medea}
alignments = find_line_to_line_alignments(target_lines, source_lines_dict, smith_waterman.align, 3, line_form_func)
save_alignments(alignments, alignments_outfile)
save_alignments_csv(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)
save_alignments_html(alignments, target_lines, source_lines_dict, alignments_outfile, lambda line:line.text_raw.split(' '), 4)


#####################################################################
## word-based line-to-line alignments on Toullier's alignment data ##
#####################################################################
smith_waterman = SmithWaterman(match_score=2,
                               mismatch_score=-1,
                               gap_score=-1,
                               n_max_alignments=1,
                               min_score_treshold=0)
line_form_func = lambda line:line.text_no_diacritics.split(' ')
alignments_outfile = 'alignments_word-based_Toullier'
target_lines = lines_chr_pat
source_lines_dict = {lines_medea[0].work_id:lines_medea}
alignments = find_line_to_line_alignments(target_lines, source_lines_dict, smith_waterman.align, 3, line_form_func, line_to_line_map)
save_alignments(alignments, alignments_outfile)
save_alignments_csv(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)
save_alignments_html(alignments, target_lines, source_lines_dict, alignments_outfile, lambda line:line.text_raw.split(' '), 4)





###########################
## LOAD MORPOLOGICAL MAP ##
###########################
# morphology files from https://github.com/gcelano/MorpheusGreekUnicode
morph_xml_fnames = ['MorpheusGreekUnicode/MorpheusGreek1-319492.xml',
                    'MorpheusGreekUnicode/MorpheusGreek319493-638984.xml',
                    'MorpheusGreekUnicode/MorpheusGreek638985-958476.xml']

def load_morphology(morph_xml_fnames, with_diacritics=True):
    '''
    Load the morphology information from xml files and create a dictionary to map word forms to their lemmas.
    
    The resulting dictionary has word forms as keys, returning a list of corresponding lemma IDs (a form can map to multiple lemmata!)
    e.g. morph_dict['ακανθον'] >>> [1003]
    
    The dictionary also contains the reverse map:
    Using the lemma ID as key returns a tuple with the lemma string and a list of all forms belonging to that lemma.
    e.g. morph_dict[1003] >>> ('ακανθος', ['ακανθ-ων', 'ακανθων', 'ακανθον', 'ακανθω', 'ακανθου', 'ακανθος'])
    
    So the dictionary would look the following for this lemma:
    {'ακανθ-ων': [1003],
     'ακανθων': [1003],
     'ακανθον': [1003],
     'ακανθω': [1003],
     'ακανθου': [1003],
     'ακανθος': [1003],
     1003: ('ακανθος', ['ακανθ-ων', 'ακανθων', 'ακανθον', 'ακανθω', 'ακανθου', 'ακανθος'])}
    '''
    max_id = -1
    morph_dict = {}
    for fname in morph_xml_fnames:
        root = ET.parse(fname).getroot()
        for token in root.findall('t'):
            
            # read form, lemma and ID
            if with_diacritics:
                form = prepare_text(token.find('f').text)
                lemma = prepare_text(token.find('l').text)
            else:
                form = strip_accents(prepare_text(token.find('b').text)) # 'b'/'e' nicht nutzbar, da Akzente zur Betonung etc enthalten (9.2.19: was heißt das genau?)
                lemma = strip_accents(prepare_text(token.find('e').text))
            _id = int(token.find('d').text)
            
            # add form to dict
            if form in morph_dict: # form is already in dict
                if _id not in morph_dict[form]: # check if form is connected to this ID already (there are forms that belong to multiple IDs !!!)
                    morph_dict[form].append(_id)
            else: # form is not yet in dict
                morph_dict[form] = [_id]
                max_id = max(max_id, _id)
            
            # add ID to dict
            if _id in morph_dict: # ID already registered
                if form not in morph_dict[_id][1]: # add form to this ID
                    forms = morph_dict[_id][1] + [form]
                    morph_dict[_id] = (lemma, forms)
            else: # register this ID with lemma and form
                morph_dict[_id] = (lemma, [form])
    
    return morph_dict, max_id
morph_dict, max_id = load_morphology(morph_xml_fnames, False)

# print all characters in the corpus, including the morphology data base:
alphabet_morphology = np.array([], dtype='<U1')
for word in morph_dict.keys():
    if type(word) is str:
        alphabet_morphology = np.unique(np.hstack([alphabet_morphology, np.unique(list(word))]))
for a in alphabet_morphology:
    print('{} "{}" {}'.format(hex(ord(a)), a, unicodedata.name(a, 'not defined')))
for a in np.unique(np.hstack([alphabet_no_diacritics, alphabet_morphology])):
    print('{} "{}" {} {} {}'.format(hex(ord(a)), a, a in alphabet_no_diacritics, a in alphabet_morphology, unicodedata.name(a, 'not defined')))


def add_lemma_ids(lines, morph_dict, max_id, with_diacritics=True):
    '''
    add all word forms that are found in the 
    '''
    for line in lines:
        lemma_ids = []
        text_lemma = ''
        if with_diacritics:
            text = line.text
        else:
            text = line.text_no_diacritics
        for word in text.split(None):
            if word not in morph_dict: # add missing word to dict
                print(word)
                max_id += 1
                morph_dict[word] = [max_id]
                morph_dict[max_id] = (word, [word])
            lemma_id = morph_dict[word]
            lemma_ids.append(lemma_id)
            text_lemma += morph_dict[lemma_id[0]][0] + ' '
        line.lemma_ids = lemma_ids
        line.text_lemma = text_lemma[:-1]        
    return max_id # note: other than lists and dictionaries, integers are immutable so it isn changes outside the function, thus return it
max_id = add_lemma_ids(lines_chr_pat, morph_dict, max_id)
max_id = add_lemma_ids(lines_medea, morph_dict, max_id)
# many words (~1000) not found in morpheus!!!



##########################################
## Line-to-line alignments on lemma ids ##
##########################################
string_mapping_function = lambda _id:morph_dict[_id][0]+' '
smith_waterman = SmithWaterman(match_score=4,
                               mismatch_score=-1,
                               gap_score=-1,
                               n_max_alignments=1,
                               min_score_treshold=5,
                               string_mapping_function=string_mapping_function)
line_form_func = lambda line:line.lemma_ids
alignments_outfile = 'alignments_lemma-based.csv'
target_lines = lines_chr_pat
source_lines_dict = {lines_medea[0].work_id:lines_medea}
alignments = find_line_to_line_alignments(target_lines, source_lines_dict, smith_waterman.align, 3, line_form_func)
save_alignments(alignments_outfile)
save_alignments_csv(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)
save_alignments_html(alignments, target_lines, source_lines_dict, alignments_outfile, line_form_func)





#################################################
## FIND WORD-BASED ALIGNMENTS USING MORPHOLOGY ##
#################################################
n_cp = len(lines_chr_pat)
k=3
min_score=2
match = 4
mismatch=-1
gap=-1
works = {lines_medea[0][0]:lines_medea}
#          ,lines_bac[0][0]:lines_bac}
alignments_outfile = 'alignments_morph.csv'
v_i = 6 # position of verse in lines_XX (this IDs in this case)
alignments = []
print('searching all verses for alignments...')
for i,v_cp in enumerate(lines_chr_pat):
    if i%10==0:
        print('{0}/{1} ({2:3.0f}%)'.format(i,n_cp, i/float(n_cp)*100))
    alignments_i = []
    for _id, verses in works.items():
        for j,v in enumerate(verses):
            a = align(v[v_i], v_cp[v_i], match, mismatch, gap, 1, min_score, False, False, lambda _id:morph_dict[_id][0]+' ')
            if a:
                heapq.heappush(alignments_i, (-a[0][0],_id)+a[0][1:]+(v[1],)) # negative score to sort descending order
    alignments_i = [(-a[0],)+a[1:] for a in alignments_i] # make score positive again
    alignments.append(alignments_i[:k])

with codecs.open(alignments_outfile, 'w', encoding='utf-8') as fid:
    fid.write(u'Vers-ID. CP;Versnr. CP; Sprecher; Vers CP')
    for _ in range(k): fid.write(u';Werk;Vers-ID Werk; Vers Werk; Alignment; Score; Start CP; End CP; Start Werk; End Werk')
    fid.write(u'\n')
    for i,(v_cp,al) in enumerate(zip(lines_chr_pat,alignments)):
        fid.write(u'{0};{1};"{2}";"{3}"'.format(v_cp[1],v_cp[2],('' if v_cp[3] is None else v_cp[3]),v_cp[4]))
        for j in range(k):
            if j<len(al):
                fid.write(u';"{0}";{1};"{2}";"{3}";{4:.0f};{5};{6};{7};{8}'.format(al[j][1],
                                                                                   al[j][5]+1,
                                                                                   works[al[j][1]][al[j][5]][4],
                                                                                   al[j][3],
                                                                                   al[j][0],
                                                                                   al[j][2][0][1]+1,
                                                                                   al[j][2][-1][1]+1,
                                                                                   al[j][2][0][0]+1,
                                                                                   al[j][2][-1][0]+1,))
            else:
                fid.write(u';;;;;;;;;')        
        fid.write(u'\n')



####################################
## FIND ALIGNMENTS FOR EACH VERSE ##
####################################
n_cp = len(lines_chr_pat)
alignments = []
k=3
min_score=14
match = 2
mismatch=-1
gap=-2
works = {lines_medea[0][0]:lines_medea}

print('searching all verses for alignments...')
for i,v_cp in enumerate(lines_chr_pat):
    if i%10==0:
        print('{0}/{1} ({2:3.0f}%)'.format(i,n_cp, i/float(n_cp)*100))
    alignments_i = []
    for _id, verses in works.iteritems():
        for j,v in enumerate(verses):
            a = align(v[4], v_cp[4], match, mismatch, gap, 1, min_score, False, False)
            if a:
                heapq.heappush(alignments_i, (-a[0][0],_id)+a[0][1:]+(v[1],)) # negative score to sort descending order
    alignments_i = [(-a[0],)+a[1:] for a in alignments_i] # make score positive again
    alignments.append(alignments_i[:k])

with codecs.open('alignments.csv', 'w', encoding='utf-8') as fid:
    fid.write(u'Vers-ID. CP;Versnr. CP; Sprecher; Vers CP')
    for _ in range(k): fid.write(u';Werk;Vers-ID Werk; Vers Werk; Alignment; Score; Start CP; End CP; Start Werk; End Werk')
    fid.write(u'\n')
    for i,(v_cp,al) in enumerate(zip(lines_chr_pat,alignments)):
        fid.write(u'{0};{1};"{2}";"{3}"'.format(v_cp[1],v_cp[2],('' if v_cp[3] is None else v_cp[3]),v_cp[4]))
        for j in range(k):
            if j<len(al):
                fid.write(u';"{0}";{1};"{2}";"{3}";{4:.0f};{5};{6};{7};{8}'.format(al[j][1],
                                                                                   al[j][5]+1,
                                                                                   works[al[j][1]][al[j][5]][4],
                                                                                   al[j][3],
                                                                                   al[j][0],
                                                                                   al[j][2][0][1]+1,
                                                                                   al[j][2][-1][1]+1,
                                                                                   al[j][2][0][0]+1,
                                                                                   al[j][2][-1][0]+1,))
            else:
                fid.write(u';;;;;;;;;')        
        fid.write(u'\n')




###########################################
## FIND ALIGNMENTS FOR GIVEN VERSE PAIRS ##
###########################################
fname_verse_map = 'Christos.paschon.1.2a.csv'
verse_map = np.loadtxt(fname_verse_map, int, delimiter='\t', skiprows=1, usecols=[0,1,2])
quellen_namen = ['Agamemnon',
                 'Prometheus',
                 'Alkestis',
                 'Andromache',
                 'Bakchen',
                 'Hekabe',
                 'Helena',
                 'Hippolytos',
                 'Iphigenie in Aulis',
                 'Iphigenie in Tauris',
                 'Medea',
                 'Orestes',
                 'Ph\"onikerinnen',
                 'Rhesos',
                 'Troerinnen',
                 'Ilias',
                 'Alexandra']
# make map dict
# for 
n_cp = len(lines_chr_pat)
alignments = []
k=3
min_score=14
match = 2
mismatch=-1
gap=-2
works = {lines_medea[0][0]:lines_medea}

print('searching all verses for alignments...')
for i,v_cp in enumerate(lines_chr_pat):
    if i%10==0:
        print('{0}/{1} ({2:3.0f}%)'.format(i,n_cp, i/float(n_cp)*100))
    alignments_i = []
    for _id, verses in works.iteritems():
        for j,v in enumerate(verses):
            a = align(v[4], v_cp[4], match, mismatch, gap, 1, min_score, False, False)
            if a:
                heapq.heappush(alignments_i, (-a[0][0],_id)+a[0][1:]+(v[1],)) # negative score to sort descending order
    alignments_i = [(-a[0],)+a[1:] for a in alignments_i] # make score positive again
    alignments.append(alignments_i[:k])

with codecs.open('alignments.csv', 'w', encoding='utf-8') as fid:
    fid.write(u'Vers-ID. CP;Versnr. CP; Sprecher; Vers CP')
    for _ in range(k): fid.write(u';Werk;Vers-ID Werk; Vers Werk; Alignment; Score; Start CP; End CP; Start Werk; End Werk')
    fid.write(u'\n')
    for i,(v_cp,al) in enumerate(zip(lines_chr_pat,alignments)):
        fid.write(u'{0};{1};"{2}";"{3}"'.format(v_cp[1],v_cp[2],('' if v_cp[3] is None else v_cp[3]),v_cp[4]))
        for j in range(k):
            if j<len(al):
                fid.write(u';"{0}";{1};"{2}";"{3}";{4:.0f};{5};{6};{7};{8}'.format(al[j][1],
                                                                                   al[j][5]+1,
                                                                                   works[al[j][1]][al[j][5]][4],
                                                                                   al[j][3],
                                                                                   al[j][0],
                                                                                   al[j][2][0][1]+1,
                                                                                   al[j][2][-1][1]+1,
                                                                                   al[j][2][0][0]+1,
                                                                                   al[j][2][-1][0]+1,))
            else:
                fid.write(u';;;;;;;;;')        
        fid.write(u'\n')




