import elo_tennis

elo = elo_tennis.EloTennis('./atp_data_demo/')
elo.update_ratings()
print(f"Prob that Fed beats Nadal on clay: {elo.predict('Roger Federer', 'Rafael Nadal', 'clay')}")
print(f"Prob that Fed beats Nadal on hard: {elo.predict('Roger Federer', 'Rafael Nadal', 'hard')}")
print(f"Prob that Fed beats Nadal on grass: {elo.predict('Roger Federer', 'Rafael Nadal', 'grass')}")
elo.plot_player_rating(['Roger Federer', 'Rafael Nadal', 'Novak Djokovic', 'Andy Murray'], 20040101, write=False)
