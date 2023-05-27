import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import datetime
from AccoMontage import set_premises, load_lead_sheet, phrase_selection, re_harmonization
import warnings
warnings.filterwarnings("ignore")




if __name__ == '__main__': 

    """
    Inference Script of AccoMontage 

    To run inference with AccoMontage, you should specify the following:
    Required:
        SONG_NAME & SONG_ROOT
            -- directory to a MIDI lead sheet file. This MIDI file should consists of two tracks, each containing melody (monophonic) and chord (polyphonic). Now complex chords (9th, 11th, and more) is supported.
        SEGMENTATION
            -- phrase annotation (string) of the MIDI file. For example, for an AABB song with 8 bars for each phrase, the annotation should be in the format 'A8A8B8B8\n'. Note that by default we only support the transition among 4-bar, 6-bar, and 8-bar phrases
        NOTE_SHIFT
            -- The number of upbeats in the  pickup bar (can be float). If no upbeat, specify 0.
    Optional:
        SPOTLIGHT
            -- a list of names of your prefered reference songs. See all 860 supported reference songs (Chinese POP) at ./data files/POP909 4bin quntization/four_beat_song_index.
        PREFILTER
            -- a tuple (a, b) controlling rhythmic patters. a, b can be integers in [0, 4], each controlling horrizontal rhythmic density and vertical voice number. Ther higher number, the denser rhythms.

    """




    """
    Configurations upon inference
    """
    #SPOTLIGHT = ['一曲红尘', '向天再借五百年', '夜曲', '我只在乎你', '撕夜', '放生', '明天你是否依然爱我', '映山红', '浪人情歌', '海芋恋', '狂浪生', '用心良苦', '男孩', '祝我生日快乐', '背对背拥抱', '舍得', '葫芦娃', '香水', '小乌龟']

    #SPOTLIGHT = ['修炼爱情']
    
    #SPOTLIGHT = ['小龙人']

    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'Boggy Brays.mid', 'A8A8B8B8\n', 0
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'Cuillin Reel.mid', 'A4A4B8B8\n', 1
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = "Kitty O'Niel's Champion.mid", 'A4A4B4B4A4A4B4B4\n', 1
    SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'Castles in the Air.mid', 'A8A8B8B8\n', 1
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'AULD LANG SYNE.mid', 'A4B4C4B4A4B4C4B4\n', 0
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'AULD LANG SYNE.mid', 'A8B8A8B8\n', 0
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = "Proudlocks's Variation.mid", 'A8A8B8B8\n', 1
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'ECNU University Song.mid', 'A8A8B8B8C8D8E4F6A8A8B8B8C8D8E4F6\n', 0
    SONG_ROOT='./demo/demo lead sheets'


    SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'lead sheet.mid', 'A8B8A8B8\n', 1
    SONG_ROOT='C:/Users/zhaoj/Desktop/for ids open house/Auld Lang Syne'


    SPOTLIGHT = []
    PREFILTER = (4, 2)
    
    #SONG_NAME, SEGMENTATION, NOTE_SHIFT = 'LetItBe.mid', 'i4A8B4A8B8C4A8B4A8B8B4C2\n', 0
    #SONG_ROOT='C:/Users/zhaoj/Desktop/for ids open house/Let It Be'

    phrase_data_dir = './data files/phrase_data.npz'
    edge_weights_dir = './data files/edge_weights.npz'
    checkpoint_dir = './data files/model_master_final.pt'
    pop909_meta_dir = './data files/pop909_quadraple_meters_index'

    print('Loading reference data (texture donors) from POP909. This may takes several seconds of time ...')
    model, acc_pool, reference_check, params = pre_liminary = set_premises(phrase_data_dir, edge_weights_dir, checkpoint_dir, pop909_meta_dir)

    lead_sheet, chord_roll, phrase_label = load_lead_sheet(SONG_ROOT, SONG_NAME, SEGMENTATION, NOTE_SHIFT)

    print(f'Phrase selection begins: {len(phrase_label)} phrases in total\
            \n\t Refer to {SPOTLIGHT} as much as possible\
            \n\t Set note density filter: {PREFILTER}')
    selection = phrase_selection(lead_sheet, phrase_label, reference_check, acc_pool, *params, PREFILTER, SPOTLIGHT)

    midi = re_harmonization(lead_sheet, chord_roll, phrase_label, *selection, model, acc_pool)
    
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = './demo/demo_generate/' + time + '.mid' 
    midi.write(save_path)
    print('Result saved at', save_path)
