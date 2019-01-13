from __future__ import division
from bs4 import BeautifulSoup
from collections  import defaultdict
from datetime import date
from StringIO import StringIO
import requests
import csv
import pandas as pd # pip install pandas
import time
import os


CSV_EXPORT = 'final.csv'


def get_football_matches(league):

	cols_matches = ["Date", "Home", "Away", "Result"]
	# Create struct like
	_cols_m = dict(zip(cols_matches, [1, 2, 3, 4]))

	all_matches = []

	page = requests.get("http://footballresults.org/league.php?all=1&league=" + league).text
	
	soup = BeautifulSoup(page, "html.parser")

	matches = soup.findAll("table", attrs={"class" : "borrsoft" })[5] #Magic number to find the correct table in html
	for match in matches.findAll("tr")[1:]:
		match = match.findAll("td")
		if match[0].text.strip() == "": continue

		new_m = dict()
		new_m["Date"] = match[_cols_m["Date"]].text.encode("UTF-8")
		new_m["Home"] = match[_cols_m["Home"]].text.encode("UTF-8")
		new_m["Away"] = match[_cols_m["Away"]].text.encode("UTF-8")
		new_m["Result"] = [ int(i) for i in match[_cols_m["Result"]].text.encode("UTF-8").split("-") ]

		all_matches.append(new_m)

	print "[+] Done fetching data from footballresults"
	# Flip to get the chronological events in the correct order
	return all_matches[::-1]


# 1 win 0 draw -1 lose
def create_record(team, scoreboard, date, result):
	match = dict()
	# Needed for elo finding in the correct timeframe
	match["Team"] = team
	match["Date"] = date

	match["GfA"] = scoreboard[team]["GfA"]
	match["GaA"] = scoreboard[team]["GaA"]
	
	if scoreboard[team]["Pld"] != 0:
		match["Pts_div_pld"] = scoreboard[team]["Pts"] / scoreboard[team]["Pld"]
	else:
		match["Pts_div_pld"] = 0

	match["Result"] = result
	
	return match


def simulate_league_scoreboard(matches):
	columns_scrbrd = ["Team", "Pld", "Gf", "Ga", "Pts", "GfA", "GaA"]
	scoreboard = defaultdict(int)
	
	columns_NN_input = ["Pts_div_pld", "Gfa", "GaA"]
	all_match_team_labeled_stats = []
	
	# m = match
	for m in matches:
		if m["Home"] not in scoreboard:
			scoreboard[m["Home"]] = defaultdict(int)

		if m["Away"] not in scoreboard:
			scoreboard[m["Away"]] = defaultdict(int)

		# Add matches
			# Find result of the match
		if m["Result"][0] == m["Result"][1]:
			all_match_team_labeled_stats.append(create_record(m["Home"], scoreboard, m["Date"], 1))
			all_match_team_labeled_stats.append(create_record(m["Away"], scoreboard, m["Date"], 1))
			scoreboard[m["Away"]]["Pts"] += 1
			scoreboard[m["Home"]]["Pts"] += 1

		elif   m["Result"][0] < m["Result"][1]:		 
			all_match_team_labeled_stats.append(create_record(m["Home"], scoreboard, m["Date"], 0))
			all_match_team_labeled_stats.append(create_record(m["Away"], scoreboard, m["Date"], 2))
			scoreboard[m["Away"]]["Pts"] += 3
			
		else:
			all_match_team_labeled_stats.append(create_record(m["Home"], scoreboard, m["Date"], 2))
			all_match_team_labeled_stats.append(create_record(m["Away"], scoreboard, m["Date"], 0))
			scoreboard[m["Home"]]["Pts"] += 3
			

		# Update scoreboard
			# Away Team
		scoreboard[m["Away"]]["Pld"] += 1
		scoreboard[m["Away"]]["Gf"] += m["Result"][1]
		scoreboard[m["Away"]]["Ga"] += m["Result"][0]
		scoreboard[m["Away"]]["GfA"] = scoreboard[m["Away"]]["Gf"] / scoreboard[m["Away"]]["Pld"]
		scoreboard[m["Away"]]["GaA"] = scoreboard[m["Away"]]["Ga"] / scoreboard[m["Away"]]["Pld"]

			# home Team
		scoreboard[m["Home"]]["Pld"] += 1
		scoreboard[m["Home"]]["Gf"] += m["Result"][0]
		scoreboard[m["Home"]]["Ga"] += m["Result"][1]
		scoreboard[m["Home"]]["GfA"] = scoreboard[m["Home"]]["Gf"] / scoreboard[m["Home"]]["Pld"]
		scoreboard[m["Home"]]["GaA"] = scoreboard[m["Home"]]["Ga"] / scoreboard[m["Home"]]["Pld"]



	# for team in scoreboard:
	# 	print team,"#######", scoreboard[team]
	return all_match_team_labeled_stats		


def find_elo(all_match_team_labeled_stats):
	# this can fit for every football league 
	#Need to implement some smart parsing of league data
	footballresults = ["Liverpool", "Man._City", "Tottenham", "Chelsea", "Arsenal", "Man._Utd.", "Leicester", "Watford", "Wolves", "West_Ham", "Everton", "Bournemouth", "Brighton", "Crystal_Palace", "Newcastle", "Burnley", "Cardiff", "Southampton", "Fulham", "Huddersfield"]
	clubelo = ["Liverpool", "ManCity", "Tottenham", "Chelsea", "Arsenal", "ManUnited", "Leicester", "Watford", "Wolves", "WestHam", "Everton", "Bournemouth", "Brighton", "CrystalPalace", "Newcastle", "Burnley", "Cardiff", "Southampton", "Fulham", "Huddersfield"]
	data = []
	to_get = []

	for club in clubelo:
		os.system('cls' if os.name == 'nt' else 'clear')
		print "[+] Done fetching data from footballresults"
		print "[!] Fetching elo data for %s" % club 
		csv_elos = requests.get("http://api.clubelo.com/" + club)

		if len(csv_elos.text.split("\n")) == 4 or csv_elos.status_code != 200: 
			print "Error downloading data. Exiting... " 
			exit(1)

		csv_elos = csv_elos.text
		f = StringIO(csv_elos)
		team_hist = pd.read_csv(f, delimiter=',', usecols=["Elo", "From", "To"] )
		
		# This is for debugging in order to be sure 
		# We are download the correnct stuff 
		# print team_hist
		# a = raw_input(club)
		team_hist["From"] =  pd.to_datetime(team_hist["From"], format="%Y-%m-%d")
		team_hist["To"] = pd.to_datetime(team_hist["To"], format="%Y-%m-%d")
		
		data.append(team_hist)
	
	print "[+] Done fetching data from clubelo"

	for team_match in  all_match_team_labeled_stats:
		elo_date = pd.to_datetime(team_match["Date"], format="%d.%m.%Y")
		team_hist = data[footballresults.index(team_match["Team"])]
		mask =  (team_hist["From"] <= elo_date) & (elo_date <= team_hist["To"] )
		team_match["Elo"] = float(team_hist.loc[mask]["Elo"])

	
	return  all_match_team_labeled_stats

def correct_format_and_write(to_write):
	for i in to_write:
		del i["Date"]
		del i["Team"]

		i["GfA"] = float(i["GfA"])
		i["GaA"] = float(i["GaA"])
		i["Elo"] = float(i["Elo"])


	keys = to_write[0].keys()
	with open(CSV_EXPORT, 'wb') as out_f:
		dict_writer = csv.DictWriter(out_f, keys)
		dict_writer.writeheader()
		dict_writer.writerows(to_write)

	print "Data written: ", len(to_write)


if __name__ == "__main__":
	# simulate_league_scoreboard(get_football_matches("EngPrem"))
	matches = simulate_league_scoreboard(get_football_matches("EngPrem"))

	semi_final = find_elo(matches)

	correct_format_and_write(semi_final)
