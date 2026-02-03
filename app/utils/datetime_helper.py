from datetime import date, datetime
from typing import Optional
from persiantools.jdatetime import JalaliDate, JalaliDateTime

def to_jalali_date(dt: Optional[date]) -> Optional[str]:
    if not dt:
        return None
    return JalaliDate(dt).strftime("%Y-%m-%d")

def to_jalali_datetime(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return JalaliDateTime(dt).strftime("%Y-%m-%d %H:%M:%S")
