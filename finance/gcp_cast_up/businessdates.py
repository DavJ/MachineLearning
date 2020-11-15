import datetime
import holidays

ONE_DAY = datetime.timedelta(days=1)
HOLIDAYS_US = holidays.US()

def next_business_day(day):
    current_date = datetime.date.fromisoformat(day)
    next_date = current_date + ONE_DAY
    while  next_date.weekday() in holidays.WEEKEND or next_date in HOLIDAYS_US:
        next_date += ONE_DAY
    return next_date.isoformat()
