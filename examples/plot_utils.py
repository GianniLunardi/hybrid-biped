import matplotlib as mpl

DEFAULT_FONT_SIZE = 20
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 6
DEFAULT_FONT_FAMILY = 'sans-serif'
DEFAULT_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_FONT_SERIF = ['Times New Roman', 'Times', 'Bitstream Vera Serif', 'DejaVu Serif', 'New Century Schoolbook',
                      'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Palatino',
                      'Charter', 'serif']
DEFAULT_FIGURE_FACE_COLOR = 'white'  # figure facecolor 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ['png']  # 'pdf'] #['png'] #,'eps']
FIGURES_DPI = 150
SHOW_LEGENDS = False
LEGEND_ALPHA = 0.5
SHOW_FIGURES = False
FIGURE_PATH = './'
LINE_WIDTH_RED = 0  # reduction of line width when plotting multiple lines on same plot
LINE_WIDTH_MIN = 1
BOUNDS_COLOR = 'silver'


mpl.rcdefaults()
mpl.rcParams['lines.linewidth'] = DEFAULT_LINE_WIDTH
mpl.rcParams['lines.markersize'] = DEFAULT_MARKER_SIZE
mpl.rcParams['patch.linewidth'] = 1
mpl.rcParams['axes.grid'] = True
mpl.rcParams['font.family'] = DEFAULT_FONT_FAMILY
mpl.rcParams['font.size'] = DEFAULT_FONT_SIZE
mpl.rcParams['font.serif'] = DEFAULT_FONT_SERIF
mpl.rcParams['text.usetex'] = DEFAULT_TEXT_USE_TEX
mpl.rcParams['axes.labelsize'] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams['legend.fontsize'] = DEFAULT_LEGEND_FONT_SIZE
mpl.rcParams['figure.facecolor'] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams['figure.figsize'] = 23, 12  # 12, 9 #