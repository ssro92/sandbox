from datetime import datetime, timedelta

def generate_date_list(start_date: str, end_date: str) -> list:
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)

    return date_list

if __name__ == "__main__":
    date_list = generate_date_list('2024-01-01', '2024-01-30')
    print(date_list)