"""
Fetch ENTSO-E Balancing Data for Greece

I download all available balancing market data:
- aFRR (Automatic Frequency Restoration Reserve) prices and volumes
- mFRR (Manual Frequency Restoration Reserve) prices and volumes
- Imbalance prices
- Accepted aggregated offers (A84) - mFRR clearing prices
- Procured balancing capacity
- Contracted reserves

Data is saved to separate CSV files for use in aFRR/mFRR bidding optimization.
"""

import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.mappings import PSRTYPE_MAPPINGS, DOCSTATUS, BSNTYPE

# I load the API key from .env file
load_dotenv()
API_KEY = os.getenv('ENTSOE_API_KEY')

if not API_KEY:
    print("ERROR: ENTSOE_API_KEY not found in .env file")
    sys.exit(1)

# I initialize the client
client = EntsoePandasClient(api_key=API_KEY)

# Greece country code
COUNTRY = 'GR'
TZ = 'Europe/Athens'

# I set the date range - last 2 years of data
END = pd.Timestamp.now(tz=TZ).normalize()
START = END - pd.DateOffset(years=2)

# Output directory
OUTPUT_DIR = Path(__file__).parent / 'balancing'
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_with_retry(func, *args, max_retries=3, **kwargs):
    """I wrap API calls with retry logic for robustness."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"    Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                print(f"    Skipping after {max_retries} attempts")
                return None
    return None


def fetch_in_chunks(func, country, start, end, chunk_months=3, **kwargs):
    """I fetch data in chunks to avoid API timeouts on large requests."""
    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + pd.DateOffset(months=chunk_months), end)
        print(f"    Fetching {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")

        data = fetch_with_retry(func, country, start=current_start, end=current_end, **kwargs)
        if data is not None and len(data) > 0:
            all_data.append(data)

        current_start = current_end

    if all_data:
        return pd.concat(all_data)
    return None


# I define ENTSO-E API constants for direct queries
ENTSOE_BASE_URL = "https://web-api.tp.entsoe.eu/api"
GREECE_DOMAIN = "10YGR-HTSO-----Y"


def fetch_accepted_offers_a84(start: pd.Timestamp, end: pd.Timestamp,
                               business_type: str = 'A97') -> pd.DataFrame:
    """
    I fetch A84 (Accepted Aggregated Offers) directly from ENTSO-E API.

    This gives us mFRR clearing prices that the entsoe-py library doesn't support.

    Parameters:
        start: Start timestamp
        end: End timestamp
        business_type: A95 for aFRR, A97 for mFRR

    Returns:
        DataFrame with columns: timestamp, direction, price
    """
    all_records = []
    current = start

    while current < end:
        # I fetch in 1-day chunks to avoid timeouts
        chunk_end = min(current + pd.DateOffset(days=1), end)

        # I format timestamps for ENTSO-E API (YYYYMMDDHHmm in UTC)
        start_str = current.tz_convert('UTC').strftime('%Y%m%d%H%M')
        end_str = chunk_end.tz_convert('UTC').strftime('%Y%m%d%H%M')

        params = {
            'documentType': 'A84',
            'controlArea_domain': GREECE_DOMAIN,
            'processType': 'A16',
            'businessType': business_type,
            'securityToken': API_KEY,
            'PeriodStart': start_str,
            'PeriodEnd': end_str
        }

        try:
            response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

            if response.status_code == 200:
                records = parse_a84_xml(response.text, current.tzinfo)
                all_records.extend(records)
            elif response.status_code == 400:
                # No data for this period
                pass
            else:
                print(f"    Warning: HTTP {response.status_code} for {current.date()}")

        except Exception as e:
            print(f"    Error fetching {current.date()}: {e}")

        current = chunk_end

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    return None


def parse_a84_xml(xml_text: str, tz) -> list:
    """I parse the A84 XML response into a list of records."""
    records = []

    # I detect the namespace dynamically from the XML
    # ENTSO-E uses different versions (3:0, 4:1, etc.)
    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            # I get the flow direction (A01=Up, A02=Down)
            flow_dir_elem = ts.find('ns:flowDirection.direction', ns)
            if flow_dir_elem is None:
                continue

            direction = 'Up' if flow_dir_elem.text == 'A01' else 'Down'

            # I find the period
            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            # I get start time and resolution
            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)

            # I assume 15-minute resolution (PT15M)
            resolution = pd.Timedelta(minutes=15)

            # I iterate through points
            for point in period.findall('ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                price_elem = point.find('ns:activation_Price.amount', ns)

                if price_elem is not None:
                    price = float(price_elem.text)
                    timestamp = start_time + (position - 1) * resolution

                    records.append({
                        'timestamp': timestamp,
                        'direction': direction,
                        'price': price
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    return records


def fetch_a84_in_chunks(start: pd.Timestamp, end: pd.Timestamp,
                        business_type: str = 'A97',
                        chunk_months: int = 1) -> pd.DataFrame:
    """I fetch A84 data in monthly chunks with progress reporting."""
    all_data = []
    current_start = start

    type_name = 'mFRR' if business_type == 'A97' else 'aFRR'

    while current_start < end:
        current_end = min(current_start + pd.DateOffset(months=chunk_months), end)
        print(f"    Fetching {type_name} {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")

        data = fetch_accepted_offers_a84(current_start, current_end, business_type)
        if data is not None and len(data) > 0:
            all_data.append(data)
            print(f"      -> {len(data)} records")

        current_start = current_end

    if all_data:
        return pd.concat(all_data)
    return None


def fetch_imbalance_volumes_a24(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    I fetch A24 (Imbalance Volumes) with processType A67 from ENTSO-E API.

    This shows real-time system imbalance - how much energy the grid needs.
    Useful for predicting activation probability.

    Returns:
        DataFrame with columns: timestamp, direction, quantity_mw
    """
    all_records = []
    current = start

    while current < end:
        chunk_end = min(current + pd.DateOffset(days=1), end)

        start_str = current.tz_convert('UTC').strftime('%Y%m%d%H%M')
        end_str = chunk_end.tz_convert('UTC').strftime('%Y%m%d%H%M')

        params = {
            'documentType': 'A24',
            'area_Domain': GREECE_DOMAIN,
            'processType': 'A67',
            'securityToken': API_KEY,
            'PeriodStart': start_str,
            'PeriodEnd': end_str
        }

        try:
            response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

            if response.status_code == 200:
                records = parse_a24_xml(response.text, current.tzinfo)
                all_records.extend(records)
            elif response.status_code == 400:
                pass  # No data for this period

        except Exception as e:
            print(f"    Error fetching {current.date()}: {e}")

        current = chunk_end

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    return None


def parse_a24_xml(xml_text: str, tz) -> list:
    """I parse the A24 XML response (imbalance volumes) into records."""
    records = []

    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            flow_dir_elem = ts.find('ns:flowDirection.direction', ns)
            if flow_dir_elem is None:
                continue

            direction = 'Up' if flow_dir_elem.text == 'A01' else 'Down'

            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)
            resolution = pd.Timedelta(minutes=15)

            for point in period.findall('.//ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                qty_elem = point.find('ns:quantity', ns)

                if qty_elem is not None:
                    quantity = float(qty_elem.text)
                    timestamp = start_time + (position - 1) * resolution

                    records.append({
                        'timestamp': timestamp,
                        'direction': direction,
                        'quantity_mw': quantity
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    return records


def fetch_hydro_reservoir_a72(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    I fetch A72 (Water Reservoirs and Hydro Storage) from ENTSO-E API.

    Weekly data showing hydro reservoir levels in MWh.
    Useful for price forecasting - high hydro = cheaper balancing prices.

    Returns:
        DataFrame with columns: week_start, reservoir_mwh
    """
    start_str = start.tz_convert('UTC').strftime('%Y%m%d%H%M')
    end_str = end.tz_convert('UTC').strftime('%Y%m%d%H%M')

    params = {
        'documentType': 'A72',
        'in_Domain': GREECE_DOMAIN,
        'ProcessType': 'A16',
        'securityToken': API_KEY,
        'PeriodStart': start_str,
        'PeriodEnd': end_str
    }

    try:
        response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

        if response.status_code == 200:
            return parse_a72_xml(response.text, start.tzinfo)

    except Exception as e:
        print(f"    Error fetching hydro reservoir: {e}")

    return None


def parse_a72_xml(xml_text: str, tz) -> pd.DataFrame:
    """I parse the A72 XML response (hydro reservoir levels) into a DataFrame."""
    records = []

    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)

            # I check resolution - typically P7D (weekly) for reservoir data
            res_elem = period.find('ns:resolution', ns)
            if res_elem is not None and res_elem.text == 'P7D':
                resolution = pd.Timedelta(days=7)
            else:
                resolution = pd.Timedelta(days=7)  # Default to weekly

            for point in period.findall('.//ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                qty_elem = point.find('ns:quantity', ns)

                if qty_elem is not None:
                    quantity = float(qty_elem.text)
                    week_start = start_time + (position - 1) * resolution

                    records.append({
                        'week_start': week_start,
                        'reservoir_mwh': quantity
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    if records:
        df = pd.DataFrame(records)
        df.set_index('week_start', inplace=True)
        return df.sort_index()
    return None


def fetch_total_load_a65(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    I fetch A65 (Total Load) from ENTSO-E API.

    15-minute resolution load data showing system demand in MW.
    Useful for predicting imbalance - high load = more stress = higher prices.

    Returns:
        DataFrame with columns: timestamp, load_mw
    """
    all_records = []
    current = start

    while current < end:
        chunk_end = min(current + pd.DateOffset(days=7), end)

        start_str = current.tz_convert('UTC').strftime('%Y%m%d%H%M')
        end_str = chunk_end.tz_convert('UTC').strftime('%Y%m%d%H%M')

        params = {
            'documentType': 'A65',
            'outBiddingZone_Domain': GREECE_DOMAIN,
            'ProcessType': 'A16',
            'securityToken': API_KEY,
            'PeriodStart': start_str,
            'PeriodEnd': end_str
        }

        try:
            response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

            if response.status_code == 200:
                records = parse_a65_xml(response.text, current.tzinfo)
                all_records.extend(records)

        except Exception as e:
            print(f"    Error fetching load {current.date()}: {e}")

        current = chunk_end

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    return None


def parse_a65_xml(xml_text: str, tz) -> list:
    """I parse the A65 XML response (total load) into records."""
    records = []

    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)
            resolution = pd.Timedelta(minutes=15)

            for point in period.findall('.//ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                qty_elem = point.find('ns:quantity', ns)

                if qty_elem is not None:
                    load_mw = float(qty_elem.text)
                    timestamp = start_time + (position - 1) * resolution

                    records.append({
                        'timestamp': timestamp,
                        'load_mw': load_mw
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    return records


# I define neighboring country domains for cross-border capacity
NEIGHBOR_DOMAINS = {
    'TR': '10YTR-TEIAS----W',   # Turkey
    'AL': '10YAL-KESH-----5',   # Albania
    'MK': '10YMK-MEPSO----8',   # North Macedonia
    'IT': '10YIT-GRTN-----B',   # Italy
    'BG': '10YCA-BULGARIA-R',   # Bulgaria
}


def fetch_cross_border_capacity_a61(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    I fetch A61 (Forecasted Transfer Capacities) for all Greece interconnections.

    Returns NTC (Net Transfer Capacity) in MW for imports and exports
    with Turkey, Albania, North Macedonia, Italy, Bulgaria.

    Useful for:
    - Understanding import/export potential
    - Cross-border balancing availability
    - Price convergence signals

    Returns:
        DataFrame with columns: timestamp, neighbor, direction, capacity_mw
    """
    all_records = []

    start_str = start.tz_convert('UTC').strftime('%Y%m%d%H%M')
    end_str = end.tz_convert('UTC').strftime('%Y%m%d%H%M')

    for country, domain in NEIGHBOR_DOMAINS.items():
        # I fetch both directions: import (neighbor→GR) and export (GR→neighbor)
        for direction, in_dom, out_dom in [
            ('import', domain, GREECE_DOMAIN),
            ('export', GREECE_DOMAIN, domain)
        ]:
            params = {
                'documentType': 'A61',
                'contract_MarketAgreement.Type': 'A01',
                'in_domain': in_dom,
                'out_domain': out_dom,
                'securityToken': API_KEY,
                'PeriodStart': start_str,
                'PeriodEnd': end_str
            }

            try:
                response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

                if response.status_code == 200:
                    records = parse_a61_xml(response.text, start.tzinfo, country, direction)
                    all_records.extend(records)

            except Exception as e:
                print(f"    Error fetching {country} {direction}: {e}")

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    return None


def parse_a61_xml(xml_text: str, tz, neighbor: str, direction: str) -> list:
    """I parse the A61 XML response (transfer capacity) into records."""
    records = []

    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)

            # I check resolution - typically PT60M (hourly) for NTC
            res_elem = period.find('ns:resolution', ns)
            if res_elem is not None and 'PT60M' in res_elem.text:
                resolution = pd.Timedelta(hours=1)
            else:
                resolution = pd.Timedelta(hours=1)

            for point in period.findall('.//ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                qty_elem = point.find('ns:quantity', ns)

                if qty_elem is not None:
                    capacity_mw = float(qty_elem.text)
                    timestamp = start_time + (position - 1) * resolution

                    records.append({
                        'timestamp': timestamp,
                        'neighbor': neighbor,
                        'direction': direction,
                        'capacity_mw': capacity_mw
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    return records


def fetch_scheduled_exchanges_a11(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    I fetch A11 (Scheduled Commercial Exchanges) for all Greece interconnections.

    Returns actual scheduled energy flows in MW between Greece and neighbors:
    Turkey, Albania, North Macedonia, Italy, Bulgaria.

    This complements A61 (NTC capacity) by showing what's actually scheduled to flow.

    Useful for:
    - Understanding real import/export patterns
    - Price spread arbitrage signals
    - System stress indicators (high imports = tight supply)
    - Balancing market prediction (net position affects imbalance)

    Returns:
        DataFrame with columns: timestamp, neighbor, direction, scheduled_mw
    """
    all_records = []

    for country, domain in NEIGHBOR_DOMAINS.items():
        # I fetch both directions: import (neighbor→GR) and export (GR→neighbor)
        for direction, in_dom, out_dom in [
            ('import', domain, GREECE_DOMAIN),
            ('export', GREECE_DOMAIN, domain)
        ]:
            records = _fetch_a11_single_direction(
                start, end, country, direction, in_dom, out_dom
            )
            all_records.extend(records)

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    return None


def _fetch_a11_single_direction(start: pd.Timestamp, end: pd.Timestamp,
                                 country: str, direction: str,
                                 in_dom: str, out_dom: str) -> list:
    """
    I fetch A11 data for a single country/direction pair in daily chunks.

    Using daily chunks because ENTSO-E API can timeout on larger requests.
    """
    all_records = []
    current = start

    while current < end:
        # I use daily chunks to avoid API timeouts
        chunk_end = min(current + pd.DateOffset(days=1), end)

        start_str = current.tz_convert('UTC').strftime('%Y%m%d%H%M')
        end_str = chunk_end.tz_convert('UTC').strftime('%Y%m%d%H%M')

        params = {
            'documentType': 'A11',
            'in_domain': in_dom,
            'out_domain': out_dom,
            'securityToken': API_KEY,
            'PeriodStart': start_str,
            'PeriodEnd': end_str
        }

        try:
            response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

            if response.status_code == 200:
                records = parse_a11_xml(
                    response.text, current.tzinfo, country, direction
                )
                all_records.extend(records)
            elif response.status_code == 400:
                # No data for this period - this is normal for some interconnections
                pass
            else:
                print(f"    Warning: HTTP {response.status_code} for {country} {direction} {current.date()}")

        except Exception as e:
            print(f"    Error fetching {country} {direction} {current.date()}: {e}")

        current = chunk_end

    return all_records


def parse_a11_xml(xml_text: str, tz, neighbor: str, direction: str) -> list:
    """
    I parse the A11 XML response (scheduled commercial exchanges) into records.

    A11 documents contain TimeSeries with:
    - in_Domain.mRID / out_Domain.mRID: The bidding zones
    - Period with timeInterval and resolution (typically PT60M for hourly)
    - Point elements with position and quantity (scheduled MW)
    """
    records = []

    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)

            # I check resolution - typically PT60M (hourly) for scheduled exchanges
            res_elem = period.find('ns:resolution', ns)
            if res_elem is not None:
                res_text = res_elem.text
                if 'PT60M' in res_text:
                    resolution = pd.Timedelta(hours=1)
                elif 'PT30M' in res_text:
                    resolution = pd.Timedelta(minutes=30)
                elif 'PT15M' in res_text:
                    resolution = pd.Timedelta(minutes=15)
                else:
                    resolution = pd.Timedelta(hours=1)  # Default to hourly
            else:
                resolution = pd.Timedelta(hours=1)

            for point in period.findall('.//ns:Point', ns):
                position_elem = point.find('ns:position', ns)
                qty_elem = point.find('ns:quantity', ns)

                if position_elem is not None and qty_elem is not None:
                    position = int(position_elem.text)
                    scheduled_mw = float(qty_elem.text)
                    timestamp = start_time + (position - 1) * resolution

                    records.append({
                        'timestamp': timestamp,
                        'neighbor': neighbor,
                        'direction': direction,
                        'scheduled_mw': scheduled_mw
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    return records


def fetch_scheduled_exchanges_a11_in_chunks(start: pd.Timestamp, end: pd.Timestamp,
                                             chunk_months: int = 1) -> pd.DataFrame:
    """
    I fetch A11 data in monthly chunks with progress reporting.

    Use this for fetching large historical periods to avoid memory issues
    and provide progress feedback.
    """
    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + pd.DateOffset(months=chunk_months), end)
        print(f"    Fetching scheduled exchanges {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")

        data = fetch_scheduled_exchanges_a11(current_start, current_end)
        if data is not None and len(data) > 0:
            all_data.append(data)
            print(f"      -> {len(data)} records")

        current_start = current_end

    if all_data:
        return pd.concat(all_data)
    return None


def fetch_generation_forecast_a69(start: pd.Timestamp, end: pd.Timestamp,
                                   psr_type: str = 'B16') -> pd.DataFrame:
    """
    I fetch A69 (Generation Forecasts) from ENTSO-E API.

    PSR Types:
        B16 = Solar
        B18 = Wind Offshore
        B19 = Wind Onshore
        B12 = Hydro Run-of-river

    Useful for predicting imbalances - high renewables = more volatility.

    Returns:
        DataFrame with columns: timestamp, forecast_mw
    """
    all_records = []
    current = start

    while current < end:
        chunk_end = min(current + pd.DateOffset(days=7), end)

        start_str = current.tz_convert('UTC').strftime('%Y%m%d%H%M')
        end_str = chunk_end.tz_convert('UTC').strftime('%Y%m%d%H%M')

        params = {
            'documentType': 'A69',
            'in_Domain': GREECE_DOMAIN,
            'processType': 'A40',
            'psrType': psr_type,
            'securityToken': API_KEY,
            'PeriodStart': start_str,
            'PeriodEnd': end_str
        }

        try:
            response = requests.get(ENTSOE_BASE_URL, params=params, timeout=60)

            if response.status_code == 200:
                records = parse_a69_xml(response.text, current.tzinfo)
                all_records.extend(records)

        except Exception as e:
            print(f"    Error fetching forecast {current.date()}: {e}")

        current = chunk_end

    if all_records:
        df = pd.DataFrame(all_records)
        df.set_index('timestamp', inplace=True)
        return df.sort_index()
    return None


def parse_a69_xml(xml_text: str, tz) -> list:
    """I parse the A69 XML response (generation forecast) into records."""
    records = []

    try:
        root = ET.fromstring(xml_text)
        ns_uri = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': ns_uri} if ns_uri else {}

        for ts in root.findall('.//ns:TimeSeries', ns):
            period = ts.find('ns:Period', ns)
            if period is None:
                continue

            start_elem = period.find('ns:timeInterval/ns:start', ns)
            if start_elem is None:
                continue

            start_time = pd.Timestamp(start_elem.text).tz_convert(tz)
            resolution = pd.Timedelta(minutes=15)

            for point in period.findall('.//ns:Point', ns):
                position = int(point.find('ns:position', ns).text)
                qty_elem = point.find('ns:quantity', ns)

                if qty_elem is not None:
                    forecast_mw = float(qty_elem.text)
                    timestamp = start_time + (position - 1) * resolution

                    records.append({
                        'timestamp': timestamp,
                        'forecast_mw': forecast_mw
                    })

    except ET.ParseError as e:
        print(f"    XML parse error: {e}")

    return records


def main():
    print("=" * 70)
    print("ENTSO-E Balancing Data Fetcher for Greece")
    print("=" * 70)
    print(f"Date range: {START.strftime('%Y-%m-%d')} to {END.strftime('%Y-%m-%d')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    datasets = {}

    # 1. Imbalance Prices
    print("1. Fetching Imbalance Prices...")
    try:
        imbalance = fetch_in_chunks(
            client.query_imbalance_prices,
            COUNTRY, START, END
        )
        if imbalance is not None:
            datasets['imbalance_prices'] = imbalance
            print(f"   -> {len(imbalance)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 2. Activated Balancing Energy Prices
    print("\n2. Fetching Activated Balancing Energy Prices...")
    try:
        activated_prices = fetch_in_chunks(
            client.query_activated_balancing_energy_prices,
            COUNTRY, START, END
        )
        if activated_prices is not None:
            datasets['activated_balancing_energy_prices'] = activated_prices
            print(f"   -> {len(activated_prices)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 3. Activated Balancing Energy Volumes
    print("\n3. Fetching Activated Balancing Energy Volumes...")
    try:
        activated_volumes = fetch_in_chunks(
            client.query_activated_balancing_energy,
            COUNTRY, START, END
        )
        if activated_volumes is not None:
            datasets['activated_balancing_energy_volumes'] = activated_volumes
            print(f"   -> {len(activated_volumes)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 4. Procured Balancing Capacity - aFRR (A51)
    print("\n4. Fetching Procured Balancing Capacity - aFRR...")
    try:
        afrr_capacity = fetch_in_chunks(
            client.query_procured_balancing_capacity,
            COUNTRY, START, END,
            process_type='A51'  # aFRR
        )
        if afrr_capacity is not None:
            datasets['procured_capacity_afrr'] = afrr_capacity
            print(f"   -> {len(afrr_capacity)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 5. Procured Balancing Capacity - mFRR (A47)
    print("\n5. Fetching Procured Balancing Capacity - mFRR...")
    try:
        mfrr_capacity = fetch_in_chunks(
            client.query_procured_balancing_capacity,
            COUNTRY, START, END,
            process_type='A47'  # mFRR
        )
        if mfrr_capacity is not None:
            datasets['procured_capacity_mfrr'] = mfrr_capacity
            print(f"   -> {len(mfrr_capacity)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 6. Contracted Reserve Prices
    print("\n6. Fetching Contracted Reserve Prices...")
    try:
        reserve_prices = fetch_in_chunks(
            client.query_contracted_reserve_prices,
            COUNTRY, START, END
        )
        if reserve_prices is not None:
            datasets['contracted_reserve_prices'] = reserve_prices
            print(f"   -> {len(reserve_prices)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 7. Contracted Reserve Amounts
    print("\n7. Fetching Contracted Reserve Amounts...")
    try:
        reserve_amounts = fetch_in_chunks(
            client.query_contracted_reserve_amount,
            COUNTRY, START, END
        )
        if reserve_amounts is not None:
            datasets['contracted_reserve_amounts'] = reserve_amounts
            print(f"   -> {len(reserve_amounts)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 8. Crossborder Balancing (if available)
    print("\n8. Fetching Crossborder Balancing...")
    try:
        crossborder = fetch_in_chunks(
            client.query_crossborder_balancing,
            COUNTRY, START, END
        )
        if crossborder is not None:
            datasets['crossborder_balancing'] = crossborder
            print(f"   -> {len(crossborder)} rows")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 9. Accepted Aggregated Offers - mFRR (A84 document type)
    # I use direct API calls because entsoe-py doesn't support this
    print("\n9. Fetching Accepted Aggregated Offers - mFRR (A84)...")
    try:
        mfrr_a84 = fetch_a84_in_chunks(START, END, business_type='A97')
        if mfrr_a84 is not None:
            datasets['accepted_offers_mfrr'] = mfrr_a84
            print(f"   -> {len(mfrr_a84)} rows total")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 10. Accepted Aggregated Offers - aFRR (A84 document type)
    print("\n10. Fetching Accepted Aggregated Offers - aFRR (A84)...")
    try:
        afrr_a84 = fetch_a84_in_chunks(START, END, business_type='A95')
        if afrr_a84 is not None:
            datasets['accepted_offers_afrr'] = afrr_a84
            print(f"   -> {len(afrr_a84)} rows total")
    except Exception as e:
        print(f"   -> Error: {e}")

    # 11. Scheduled Commercial Exchanges - A11 (cross-border flows)
    # I fetch actual scheduled flows between Greece and neighbors
    print("\n11. Fetching Scheduled Commercial Exchanges - A11 (cross-border)...")
    try:
        scheduled_exchanges = fetch_scheduled_exchanges_a11_in_chunks(START, END)
        if scheduled_exchanges is not None:
            datasets['scheduled_exchanges'] = scheduled_exchanges
            print(f"   -> {len(scheduled_exchanges)} rows total")
            # I also print per-neighbor summary
            for neighbor in scheduled_exchanges['neighbor'].unique():
                neighbor_data = scheduled_exchanges[scheduled_exchanges['neighbor'] == neighbor]
                print(f"      {neighbor}: {len(neighbor_data)} records")
    except Exception as e:
        print(f"   -> Error: {e}")

    # Save all datasets
    print("\n" + "=" * 70)
    print("Saving datasets...")
    print("=" * 70)

    for name, df in datasets.items():
        filepath = OUTPUT_DIR / f'{name}.csv'
        df.to_csv(filepath)
        print(f"  Saved: {filepath.name} ({len(df)} rows)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total datasets: {len(datasets)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nFiles created:")
    for name in datasets.keys():
        print(f"  - {name}.csv")

    # Preview first dataset
    if datasets:
        first_name = list(datasets.keys())[0]
        first_df = datasets[first_name]
        print(f"\nPreview of {first_name}:")
        print(first_df.head())


if __name__ == "__main__":
    main()
