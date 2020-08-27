import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as dt
import plotly.graph_objects as go
import plotly.express as px

def sigmoid(z):
    '''The sigmoid function.'''
    return np.where(z >= 0, 
                    1 / (1 + np.exp(-z)), 
                    np.exp(z) / (1 + np.exp(z)))

def get_player_initials(player):
    ''' just returns the player's initials.
        Inputting "Roger Federer" returns "RF". '''
    return ''.join([word[0] for word in player.split(" ")])

class EloTennis:
    ''' An object of class EloTennis records the history of each player's
        ratings based on match history. This class is used both visualization
        and prediction. The hyperparameters used for the ratings are obtained 
        from previous computations. See hongsuh7.github.io for more info. '''
    
    def __init__(self, path):
        '''
            path: file path in which the math data lives. Should end with a /

            ATP data is obtained from Jeff Sackmann's github:
            https://github.com/JeffSackmann/tennis_atp
        '''
        
        # set parameters. they are fixed here
        self.k_params = np.multiply(np.ones((50, 3)), [0.47891635, 4.0213623 , 0.25232273])
        self.a_params = pd.read_csv("default_parameters.csv").to_numpy()[:50, :]
        
        # collect season data
        self.data = []
        files = os.listdir(path)
        files.sort()
        print('Read the following files in order:')
        for filename in files:
            if filename.endswith('.csv'):
                print(filename)
                self.data.append(pd.read_csv(path + filename).sort_values(by=['tourney_date', 'match_num']))
        
        # collect all the player names
        self.players = {player for player in self.data[0]['winner_name']}
        self.players = self.players.union({player for player in self.data[0]['loser_name']})
        for season in self.data:
            self.players = self.players.union({player for player in season['winner_name']})
            self.players = self.players.union({player for player in season['loser_name']})            
        
        # first axis is the number of matches (plus one) played by the player;
        # second axis is the 50 sets of hyperparameters used;
        # third axis is the surfaces.
        self.ratings = {player: np.ones((1,50,3)) for player in self.players}

        # stores the dates of matches, roughly.
        # the format is yyyymmdd in an integer, like 20130513 for may 13, 2013.
        self.dates = {player: [] for player in self.players}
    
    def k(self, n, ps):
        '''returns the vector K-factor, which dictates how sensitive ratings are
        to an individual match and depends on the number of matches played.'''
        return np.multiply(ps[:,0], 
                           np.power(ps[:,1] + n, -ps[:,2])
                          )
    
    
    def update_one(self, x, y, n1, n2, k_params, a_params, s):
        '''this function updates one match.

        x : winner ratings
        y : loser ratings
        n1 : winner number of matches played
        n2 : loser number of matches played
        k_params : parameters for k-factor function; see blog post
        a_params : parameters for surface matrix; see blog post
        s : surface, integer. Clay=1, Grass=2, Hard=3.

        Returns the prior probability that the winner wins, and the values to update 
        the winner rating and loser rating by. '''
        z = np.multiply(np.dot(a_params.reshape((len(a_params),3,3)), s), sigmoid(y-x))
        z1 = z[:,0]
        z2 = z[:,1]
        z3 = z[:,2]
        u1 = np.multiply(self.k(n1, k_params), z1)
        u2 = np.multiply(self.k(n1, k_params), z2)
        u3 = np.multiply(self.k(n1, k_params), z3)
        v1 = -np.multiply(self.k(n2, k_params), z1)
        v2 = -np.multiply(self.k(n2, k_params), z2)
        v3 = -np.multiply(self.k(n2, k_params), z3)
        
        u = np.transpose(np.array([u1,u2,u3]))
        v = np.transpose(np.array([v1,v2,v3]))
        prob = np.dot(sigmoid(x-y), s)
        return(prob, u, v)
    
    def update_ratings(self):
        '''this function updates the ratings corresponding to the default
           parameters. '''
        
        # first reset the ratings.
        self.ratings = {player: np.ones((1,50,3)) for player in self.players}
        
        n = sum([len(dat) for dat in self.data])
        counter = 0
        for i in range(len(self.data)):
            for j, row in self.data[i].iterrows():
                
                winner = row['winner_name']
                loser = row['loser_name']
                surface = row['surface']
                if surface == 'Clay':
                    s = np.array([1,0,0])
                elif surface == 'Hard':
                    s = np.array([0,0,1])
                else: # Carpet gets classified as Grass. 
                    s = np.array([0,1,0])
                
                
                # get ratings.
                wnm = len(self.ratings[winner]) - 1
                lnm = len(self.ratings[loser]) - 1
                wrating = self.ratings[winner][-1,:,:]
                lrating = self.ratings[loser][-1,:,:]
                
                # update.
                prob, u1, u2 = self.update_one(wrating, lrating, wnm, lnm, 
                                               self.k_params, self.a_params, s)
                self.ratings[winner] = np.concatenate([self.ratings[winner], (wrating + u1).reshape(1,50,3)], axis=0)
                self.ratings[loser] = np.concatenate([self.ratings[loser], (lrating + u2).reshape(1,50,3)], axis=0)

                # add date.
                date = row['tourney_date']
                self.dates[winner].append(str(date))
                self.dates[loser].append(str(date))

                if counter % (n//20) == 1:
                    print('Progress bar: %d / 100' % (100*counter//n))

                counter += 1

    def predict(self, p1, p2, s):
        ''' returns the probability that p1 beats p2 on surface s.
            This function uses the most recent ratings of p1 and p2.

            p1 : player one name
            p2 : player two name
            s : surface, integer'''
        if isinstance(s, str):
            if s.lower() == "clay":
                surface = 0
            elif s.lower() == "grass":
                surface = 1
            elif s.lower() == "hard":
                surface = 2
            else:
                print("Invalid surface. Returning 0.5.")
                return 0.5
        else:
            surface = s
        r1 = self.ratings[p1][-1,:,surface]
        r2 = self.ratings[p2][-1,:,surface]
        return np.mean(sigmoid(r1 - r2))

    def get_player_rating(self, player):
        ''' returns the player rating history with averaged ratings. '''
        return np.mean(self.ratings[player], axis=1)


    def plot_player_rating(self, players, start_date=0, write=False):
        ''' interactively plots the players' rating histories with averaged ratings,
            starting at start_date. 

            If write is False, shows the image. Otherwise, you can input a string
            which will become the file name of the html file.'''
        fig = go.Figure()
        player_buttons = []
        surface_buttons = []
        counter = 0
        min_date = 1e10
        max_date = 0
        dash = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

        for player in players:
            ratings = self.get_player_rating(player)
            indices = np.argwhere(np.array([int(d) for d in self.dates[player]], dtype=int) > start_date).flatten()
            dates = dt.datestr2num(self.dates[player])
            if min(dates) < min_date:
                min_date = min(dates)
            if max(dates) > max_date:
                max_date = max(dates)

            fig.add_trace(
                go.Scatter(
                    x = dates[indices],
                    y = ratings[(indices + 1),0],
                    name = "Clay,  " + get_player_initials(player),
                    #marker=dict(color="darkorange"),
                    opacity = 1 if counter == 0 else 0.25,
                    visible = True,
                    hoverinfo = 'none',
                    line = dict(
                            color = px.colors.qualitative.Plotly[counter],
                            width = 3
                        )
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x = dates[indices],
                    y = ratings[(indices + 1),1],
                    name = "Grass, " + get_player_initials(player),
                    #mode="lines",
                    #marker=dict(color="yellowgreen"),
                    opacity = 1 if counter == 0 else 0.25,
                    visible = False,
                    hoverinfo = 'none',
                    line = dict(
                            color = px.colors.qualitative.Plotly[counter],
                            width = 3
                        )
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x = dates[indices],
                    y = ratings[(indices + 1),2],
                    name = "Hard,  " + get_player_initials(player),
                    #mode="lines",
                    #marker=dict(color="steelblue"),
                    opacity = 1 if counter == 0 else 0.25,
                    visible = False,
                    hoverinfo = 'none',
                    line = dict(
                            color = px.colors.qualitative.Plotly[counter],
                            width = 3
                        )
                    )
                )

            opacity = [0.25 for _ in range(3*len(players))]
            opacity[(3*counter):(3*counter + 3)] = [1, 1, 1]

            player_buttons.append(
                dict(
                    label = player,
                    method = 'restyle',
                    args = [
                        {'opacity' : opacity},
                        {'title' : player + " Ratings"}
                        ]
                    )
                )

            counter += 1

        surface_buttons = [
            dict(
                label = "Clay ratings",
                method = 'restyle',
                args = [{'visible' : [(i % 3 == 0) for i in range(3*len(players))]}]
                ),
            dict(
                label = "Grass ratings",
                method = 'restyle',
                args = [{'visible' : [(i % 3 == 1) for i in range(3*len(players))]}]
                ),
            dict(
                label = "Hard ratings",
                method = 'restyle',
                args = [{'visible' : [(i % 3 == 2) for i in range(3*len(players))]}]
                )
        ]

        fig.update_layout(
            updatemenus=[
                dict(
                    type = 'buttons',
                    direction = 'down',
                    #active = 0,
                    buttons = player_buttons,
                    pad = {"r": 10, "t": 10},
                    showactive = True
                    ),
                dict(
                    type = 'buttons',
                    direction = 'down',
                    buttons = surface_buttons,
                    pad={"r": 10, "t": 10},
                    showactive = True,
                    y = 0.5
                    )
            ])

        fig.update_xaxes(range=[max(min_date, dt.datestr2num(str(start_date))), max_date])
        fig.update_yaxes(range=[0, 6.3])

        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = dt.datestr2num([str(i) + '0101' for i in range(2000, 2021)][::2]),
                ticktext = (['\'0' + str(i) for i in range(10)] + ['\'' + str(i) for i in range(10,21)])[::2]
            ),
            title_text = "Big 4 Ratings, 2004-2020"
        )

        if write:
            fig.write_html(write)
        else:
            fig.show()

