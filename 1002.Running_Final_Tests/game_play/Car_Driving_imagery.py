from code import interact
from tracemalloc import start
import numpy as np 
import cv2

from PIL import Image
import os
class Car_Image:
    def __init__(self, cfg):
        self.cfg = cfg
        here = os.getcwd() + '/game_play'
        road_tile_file = f"{here}/images/road_tile.jpg"
        self.road_tile = np.array((Image.open(road_tile_file)).resize((24,24), resample=Image.BILINEAR))[:,:,:3]

        car_on_road_tile_file = f"{here}/images/car_on_road_tile.png"
        self.car_on_road_tile = np.array((Image.open(car_on_road_tile_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

        tree_file = f"{here}/images//tree_file.png"
        self.tree_tile = np.array((Image.open(tree_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

        car_hit_tree_file = f"{here}/images/car_hit_tree.png"
        self.car_hit_tree_tile = np.array((Image.open(car_hit_tree_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

        swamp_file = f"{here}/images/swamp_file.png"
        self.swamp_tile = np.array((Image.open(swamp_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

        car_on_swamp_file = f"{here}/images/car_on_swamp.png"
        self.car_on_swamp_tile = np.array((Image.open(car_on_swamp_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

        finish_line_file = f"{here}/images/finish_line.png"
        self.finish_line_tile = np.array((Image.open(finish_line_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

        car_on_finish_line_file = f"{here}/images/car_on_finish_line.png"
        self.car_on_finish_line_tile = np.array((Image.open(car_on_finish_line_file)).resize((24,24), resample=Image.NEAREST))[:,:,:3]

    def create_board_image(self, raceworld):
        view = raceworld.observable_size[1]
        start_col = raceworld.state_coors[1]-1
        end_col = start_col + view  #non inclusive
        
        cliffs = raceworld.cliffs_idx
        holes = raceworld.holes_idx
        goals = raceworld.goals_idx
        state = raceworld.state


        board = []
        for i in range((raceworld.env_size[0]* raceworld.env_size[1])):
            col= raceworld.stateIdx_to_coors[i][1]
            if col >= start_col and col < end_col:  
                if i in cliffs:
                    if i == state:
                        board.append(self.car_hit_tree_tile)
                    else:
                        board.append(self.tree_tile)
                elif i in goals:
                    if i == state:
                        board.append(self.car_on_finish_line_tile)
                    else:
                        board.append(self.finish_line_tile)
                elif i in holes:
                    if i == state:
                        board.append(self.car_on_swamp_tile)
                    else:
                        board.append(self.swamp_tile)
                else:
                    if i == state:
                        board.append(self.car_on_road_tile)
                    else:
                        board.append(self.road_tile)

        board = np.array(board).reshape(self.cfg.observable_size[0],self.cfg.observable_size[1],24,24,3).transpose(0,2,1,3,4)
        board = board.reshape(self.cfg.observable_size[0]*24,self.cfg.observable_size[1]*24,3).astype(np.uint8)
        board = cv2.resize(board, (self.cfg.image_size[0],self.cfg.image_size[1]), interpolation=cv2.INTER_NEAREST)
        return board