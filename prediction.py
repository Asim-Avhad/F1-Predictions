import fastf1
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

fastf1.Cache.enable_cache('cache')

def get_session_data(year, gp_name, session_name):
    """
    Get lap times and telemetry data for a specific session
    """
    try:
        session = fastf1.get_session(year, gp_name, session_name)
        session.load()

        lap_times = session.laps

        valid_laps = lap_times[lap_times['LapTime'].notna()]

        driver_data = []
        
        for driver in valid_laps['Driver'].unique():
            driver_laps = valid_laps[valid_laps['Driver'] == driver]
            
            if len(driver_laps) > 0:
                fastest_lap = driver_laps['LapTime'].min()
                avg_lap = driver_laps['LapTime'].mean()
                consistency = driver_laps['LapTime'].std()
                total_laps = len(driver_laps)
                
                driver_data.append({
                    'Driver': driver,
                    'FastestLap': fastest_lap.total_seconds() if pd.notna(fastest_lap) else None,
                    'AverageLap': avg_lap.total_seconds() if pd.notna(avg_lap) else None,
                    'Consistency': consistency.total_seconds() if pd.notna(consistency) else None,
                    'TotalLaps': total_laps,
                    'Session': session_name
                })
        
        return pd.DataFrame(driver_data)
    
    except Exception as e:
        print(f"Error loading {session_name}: {e}")
        return pd.DataFrame()

def get_qualifying_positions(year, gp_name):
    """
    Get qualifying positions and times
    """
    try:
        qualifying = fastf1.get_session(year, gp_name, 'Q')
        qualifying.load()

        results = qualifying.results
        
        qual_data = []
        for _, row in results.iterrows():
            qual_data.append({
                'Driver': row['Abbreviation'],
                'QualifyingPosition': row['Position'],
                'Q1Time': row['Q1'].total_seconds() if pd.notna(row['Q1']) else None,
                'Q2Time': row['Q2'].total_seconds() if pd.notna(row['Q2']) else None,
                'Q3Time': row['Q3'].total_seconds() if pd.notna(row['Q3']) else None,
            })
        
        return pd.DataFrame(qual_data)
    
    except Exception as e:
        print(f"Error loading qualifying: {e}")
        return pd.DataFrame()

def calculate_prediction_score(practice_data, qualifying_data):
    """
    Calculate prediction score based on practice sessions and qualifying
    """
    all_practice = pd.concat(practice_data, ignore_index=True)

    driver_scores = {}
    
    for driver in all_practice['Driver'].unique():
        driver_practice = all_practice[all_practice['Driver'] == driver]
        session_weights = {'FP1': 0.15, 'FP2': 0.25, 'FP3': 0.35, 'Sprint': 0.25}

        fastest_laps = []
        weights = []
        
        for _, row in driver_practice.iterrows():
            if pd.notna(row['FastestLap']):
                fastest_laps.append(row['FastestLap'])
                weights.append(session_weights.get(row['Session'], 0.2))
        
        if fastest_laps:
            avg_fastest = np.average(fastest_laps, weights=weights)

            consistency_scores = [row['Consistency'] for _, row in driver_practice.iterrows() if pd.notna(row['Consistency'])]
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 10.0

            total_laps = driver_practice['TotalLaps'].sum()
            
            driver_scores[driver] = {
                'AvgFastestLap': avg_fastest,
                'Consistency': avg_consistency,
                'TotalLaps': total_laps,
                'PracticeScore': 0
            }
    
    if driver_scores:
        fastest_overall = min([data['AvgFastestLap'] for data in driver_scores.values()])
        
        for driver, data in driver_scores.items():
            lap_time_score = (data['AvgFastestLap'] - fastest_overall) * 1000 

            consistency_score = data['Consistency'] * 100 if data['Consistency'] else 500
            reliability_score = min(data['TotalLaps'] / 50, 1.0) 
            data['PracticeScore'] = lap_time_score + consistency_score - (reliability_score * 100)

    final_scores = []
    
    for _, qual_row in qualifying_data.iterrows():
        driver = qual_row['Driver']
        
        if driver in driver_scores:
            practice_score = driver_scores[driver]['PracticeScore']
            qualifying_pos = qual_row['QualifyingPosition']

            final_score = (qualifying_pos * 0.6) + (practice_score * 0.4 / 100)
            
            final_scores.append({
                'Driver': driver,
                'QualifyingPosition': qualifying_pos,
                'PracticeScore': practice_score,
                'FinalScore': final_score,
                'AvgFastestLap': driver_scores[driver]['AvgFastestLap'],
                'Consistency': driver_scores[driver]['Consistency']
            })
    
    return pd.DataFrame(final_scores).sort_values('FinalScore')

def predict_race_winner():
    """
    Main function to predict F1 race winner
    """
    print("🏁 F1 British GP 2025 Race Winner Prediction")
    print("=" * 50)

    year = 2025
    possible_gp_names = ["British Grand Prix"]
    
    print(f"Analyzing data for British GP {year}...")

    gp_name = None
    for name in possible_gp_names:
        try:
            test_session = fastf1.get_session(year, name, 'FP1')
            gp_name = name
            print(f"Found GP data using name: '{name}'")
            break
        except Exception as e:
            print(f"❌ '{name}' not found: {str(e)[:100]}...")
            continue
    
    if not gp_name:
        print("Could not find British GP 2025 data.")
        print("Available GPs for 2025 might be:")

        try:
            schedule = fastf1.get_event_schedule(year)
            print(schedule[['EventName', 'Location', 'EventDate']].to_string())
        except:
            print("Could not retrieve 2025 schedule")
        return None
    
    print(f"\nLoading session data for {gp_name}...")
 
    practice_sessions = ['FP1', 'FP2', 'FP3']
    practice_data = []
    
    for session in practice_sessions:
        print(f"Loading {session}...")
        data = get_session_data(year, gp_name, session)
        if not data.empty:
            practice_data.append(data)
            print(f"{session}: {len(data)} drivers loaded")
        else:
            print(f"No data available for {session}")

    print("Loading Qualifying...")
    qualifying_data = get_qualifying_positions(year, gp_name)
    
    if not qualifying_data.empty:
        print(f"Qualifying: {len(qualifying_data)} drivers loaded")
    else:
        print("No qualifying data available")
    
    if not practice_data or qualifying_data.empty:
        print("Insufficient data to make prediction")
        return None

    print("\n🔍 Analyzing performance data...")
    predictions = calculate_prediction_score(practice_data, qualifying_data)
    
    if predictions.empty:
        print("Unable to generate predictions")
        return None

    print("\n🏆 RACE WINNER PREDICTIONS")
    print("=" * 50)
    
    top_10 = predictions.head(10)
    
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        fastest_lap_str = f"{row['AvgFastestLap']:.3f}s" if pd.notna(row['AvgFastestLap']) else "N/A"
        consistency_str = f"{row['Consistency']:.3f}s" if pd.notna(row['Consistency']) else "N/A"
        
        print(f"{i:2d}. {row['Driver']:3s} | "
              f"Qual: P{row['QualifyingPosition']:2.0f} | "
              f"Avg Fast Lap: {fastest_lap_str} | "
              f"Consistency: ±{consistency_str} | "
              f"Score: {row['FinalScore']:.2f}")

    winner = predictions.iloc[0]
    print(f"\nPREDICTED WINNER: {winner['Driver']}")
    print(f"   Starting Position: P{winner['QualifyingPosition']:.0f}")
    print(f"   Average Fastest Lap: {winner['AvgFastestLap']:.3f}s")
    print(f"   Prediction Confidence: {(1 - (winner['FinalScore'] - predictions.iloc[1]['FinalScore']) / 10) * 100:.1f}%")
    
    # Top 3 podium prediction
    print(f"\n🏅 PREDICTED PODIUM:")
    print(f"   🥇 1st: {predictions.iloc[0]['Driver']}")
    print(f"   🥈 2nd: {predictions.iloc[1]['Driver']}")
    print(f"   🥉 3rd: {predictions.iloc[2]['Driver']}")
    
    print(f"\n📋 Analysis based on:")
    print(f"   • Practice Sessions: {', '.join([data['Session'].iloc[0] for data in practice_data])}")
    print(f"   • Qualifying Results")
    print(f"   • Lap Time Consistency")
    print(f"   • Session Reliability")
    
    return predictions

if __name__ == "__main__":
    try:
        print("🔍 Checking available F1 races for 2025...")
        try:
            schedule = fastf1.get_event_schedule(2025)
            print("\nAvailable F1 races for 2025:")
            print(schedule[['EventName', 'Location', 'EventDate']].head(10).to_string())
            print("\n" + "="*50)
        except Exception as e:
            print(f"Could not retrieve schedule: {e}")
        
        predictions = predict_race_winner()
        
        if predictions is None:
            print("\nAlternative: Testing with 2024 British GP data...")
            print("Note: 2025 season data might not be available yet.")

            year = 2024
            gp_name = "Great Britain"
            
            print(f"\nLoading 2024 {gp_name} GP data for demonstration...")
            
            practice_data = []
            for session in ['FP1', 'FP2', 'FP3']:
                data = get_session_data(year, gp_name, session)
                if not data.empty:
                    practice_data.append(data)
            
            qualifying_data = get_qualifying_positions(year, gp_name)
            
            if practice_data and not qualifying_data.empty:
                predictions = calculate_prediction_score(practice_data, qualifying_data)
                print("\n🏆 2024 BRITISH GP ANALYSIS (for reference)")
                print("=" * 50)
                
                top_5 = predictions.head(5)
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    print(f"{i}. {row['Driver']} - Score: {row['FinalScore']:.2f}")
                    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have fastf1 installed: pip install fastf1")
        print("Note: This script requires internet connection to fetch F1 data")