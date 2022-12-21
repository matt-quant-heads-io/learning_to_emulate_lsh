import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

width = 30
im_w = width / 5

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    
    to_Vector( name='input', n_filer=200, width=1, offset="(-5,0,0)", caption="QueryCentroidVector: \n 200$\\times $1", height=15, depth=1),
    to_Conv1DRelu( name='l1', s_filer=1, n_filer=128, caption="Conv1D:128$\\times $1; ReLu", width=2.5, height=20, depth=2.5, offset="(-0.7,0,0)"),
    to_connection( "input", "l1" ),
    to_Vector_nolabel( name='maxpooling', n_filer=1, width=1, offset="(2,0,0)", caption="Maxpooling", height=15, depth=1),
    to_connection( "l1", "maxpooling" ),
    to_Conv1DRelu( name='l2', s_filer=1, n_filer=128, offset="(4.3,0,0)", caption="Conv1D:128$\\times $1; ReLu", width=2.5, height=20, depth=2.5),
    to_connection( "maxpooling", "l2"),
    to_Conv1DRelu( name='l3', s_filer=1, n_filer=256, offset="(7.4,0,0)", caption="Conv1D:256$\\times $1; ReLu", width=2.5, height=20, depth=2.5),
    to_connection( "l2", "l3" ),
    to_Vector_nolabel( name='output', n_filer=10, offset="(9.3,0,0)", caption='softmax; \n Label', width=1, height=1, depth=1),
    to_connection( "l3", "output" ),
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()