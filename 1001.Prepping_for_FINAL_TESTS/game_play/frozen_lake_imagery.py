from code import interact
import numpy as np 
import cv2

class FL_Image:
    def __init__(self, cfg):
        self.cfg = cfg
        self.background_colour = B = [51,204,255]
        self.hole_colour = h = [0, 0, 102]
        self.man_colour = M = [0, 0, 0]
        self.uncovered_goal = g = [255,215,0]
        self.lit_up_goal = L = [255, 204, 102]
        self.border = W = [255,255,255]


        self.F = np.array([[B]*12]*12).astype(np.uint8)
        for idx,f in enumerate(self.F[0]):
            self.F[0][idx] = W
        for idx,f in enumerate(self.F[11]):
            self.F[11][idx] = W
        for f in self.F:
            f[0] = W
            f[11] = W
        self.H = np.array([
            [W,W,W,W,W,W,W,W,W,W,W,W],
            [W,B,B,h,h,h,h,h,h,B,B,W],
            [W,B,h,h,h,h,h,h,h,h,B,W],
            [W,h,h,h,h,h,h,h,h,h,h,W],
            [W,h,h,h,h,h,h,h,h,h,h,W],
            [W,h,h,h,h,h,h,h,h,h,h,W],
            [W,h,h,h,h,h,h,h,h,h,h,W],
            [W,h,h,h,h,h,h,h,h,h,h,W],
            [W,h,h,h,h,h,h,h,h,h,h,W],
            [W,B,h,h,h,h,h,h,h,h,B,W],
            [W,B,B,h,h,h,h,h,h,B,B,W],
            [W,W,W,W,W,W,W,W,W,W,W,W],
        ]
        ).astype(np.uint8)
        self.ManH = np.array([
            [W,W,W,W,W,W,W,W,W,W,W,W],
            [W,B,B,h,h,M,M,h,h,B,B,W],
            [W,B,h,h,h,M,M,h,h,h,B,W],
            [W,h,M,M,M,M,M,M,M,M,h,W],
            [W,h,h,h,h,M,M,h,h,h,h,W],
            [W,h,h,h,h,M,M,h,h,h,h,W],
            [W,h,h,h,M,M,M,M,h,h,h,W],
            [W,h,h,M,M,h,h,M,M,h,h,W],
            [W,h,M,M,h,h,h,h,M,M,h,W],
            [W,B,M,M,h,h,h,h,M,M,B,W],
            [W,B,B,h,h,h,h,h,h,B,B,W],
            [W,W,W,W,W,W,W,W,W,W,W,W],
        ]
        ).astype(np.uint8)
        self.ManF = np.array([
            [W,W,W,W,W,W,W,W,W,W,W,W],
            [W,B,B,B,B,M,M,B,B,B,B,W],
            [W,B,B,B,B,M,M,B,B,B,B,W],
            [W,B,M,M,M,M,M,M,M,M,B,W],
            [W,B,B,B,B,M,M,B,B,B,B,W],
            [W,B,B,B,B,M,M,B,B,B,B,W],
            [W,B,B,B,M,M,M,M,B,B,B,W],
            [W,B,B,M,M,B,B,M,M,B,B,W],
            [W,B,M,M,B,B,B,B,M,M,B,W],
            [W,B,M,M,B,B,B,B,M,M,B,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,W,W,W,W,W,W,W,W,W,W,W],
        ]
        ).astype(np.uint8)
        self.G = np.array([
            [W, W, W, W, W, W, W, W, W, W, W, W],
            [W, g, B, B, B, B, B, B, B, B, g, W],
            [W, B, g, B, B, B, B, B, B, g, B, W],
            [W, B, B, g, g, g, g, g, g, B, B, W],
            [W, B, B, g, g, g, g, g, g, B, B, W],
            [W, B, B, g, g, g, g, g, g, B, B, W],
            [W, B, B, g, g, g, g, g, g, B, B, W],
            [W, B, B, g, g, g, g, g, g, B, B, W],
            [W, B, B, g, g, g, g, g, g, B, B, W],
            [W, B, g, B, B, B, B, B, B, g, B, W],
            [W, g, B, B, B, B, B, B, B, B, g, W],
            [W, W, W, W, W, W, W, W, W, W, W, W],
        ]
        ).astype(np.uint8)
        self.ManG = np.array([
            [W, W, W, W, W, W, W, W, W, W, W, W],
            [W, L, L, L, L, M, M, L, L, L, L, W],
            [W, L, L, L, L, M, M, L, L, L, L, W],
            [W, L, M, M, M, M, M, M, M, M, L, W],
            [W, L, L, g, g, M, M, g, g, L, L, W],
            [W, L, L, g, g, M, M, g, g, L, L, W],
            [W, L, L, g, M, M, M, M, g, L, L, W],
            [W, L, L, M, M, g, g, M, M, L, L, W],
            [W, L, M, M, g, g, g, g, M, M, L, W],
            [W, L, M, M, L, L, L, L, M, M, L, W],
            [W, L, L, L, L, L, L, L, L, L, L, W],
            [W, W, W, W, W, W, W, W, W, W, W, W],
        ]).astype(np.uint8)

        self.ManKey = np.array([
            [W,W,W,W,W,W,W,W,W,W,W,W],
            [W,B,B,B,B,M,M,B,B,B,B,W],
            [W,B,B,B,B,M,M,B,B,B,B,W],
            [W,B,M,M,M,M,M,M,M,M,B,W],
            [W,B,g,B,g,M,M,g,g,g,g,W],
            [W,g,B,B,B,M,M,g,B,B,B,W],
            [W,B,g,B,g,M,M,M,B,B,B,W],
            [W,B,B,g,M,B,B,M,M,B,B,W],
            [W,B,M,M,B,B,B,B,M,M,B,W],
            [W,B,M,M,B,B,B,B,M,M,B,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,W,W,W,W,W,W,W,W,W,W,W],
        ]).astype(np.uint8)

        self.Key = np.array([
            [W,W,W,W,W,W,W,W,W,W,W,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,B,B,g,B,B,B,B,B,B,B,W],
            [W,B,g,B,g,g,g,g,g,g,g,W],
            [W,g,B,B,B,g,g,g,B,B,B,W],
            [W,B,g,B,g,B,B,B,B,B,B,W],
            [W,B,B,g,B,B,B,B,B,B,B,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,B,B,B,B,B,B,B,B,B,B,W],
            [W,W,W,W,W,W,W,W,W,W,W,W],
        ]).astype(np.uint8)

    def create_board_image(self, lakes, goals, state, keys=[]):
        
        board = []
        for i in range((self.cfg.env_size[0]* self.cfg.env_size[1])):
            if i in keys:
                if i == state:
                    board.append(self.ManKey)
                else:
                    board.append(self.Key)
            elif i in lakes:
                if i == state:
                    board.append(self.ManH)
                else:
                    board.append(self.H)
            elif i in goals:
                if i == state:
                    board.append(self.ManG)
                else:
                    board.append(self.G)
            else:
                if i == state:
                    board.append(self.ManF)
                else:
                    board.append(self.F)
        
        board = np.array(board).reshape(self.cfg.env_size[0],self.cfg.env_size[1],12,12,3).transpose(0,2,1,3,4)
        board = board.reshape(self.cfg.env_size[0]*12,self.cfg.env_size[1]*12,3).astype(np.uint8)
        board = cv2.resize(board, (self.cfg.image_size[0],self.cfg.image_size[1]), interpolation=cv2.INTER_AREA)
        return board