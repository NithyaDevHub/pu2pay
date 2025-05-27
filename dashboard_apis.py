import asyncio
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from Helpers.helpers import get_files_from_azure_blob_container, get_files_from_local_directory
from extract_metadata import extract_metadata, get_file_metadata_from_blob

from sqlalchemy import Numeric, create_engine, Column, Integer, Float, String, Date, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List
from datetime import date
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

MAX_FILES_PER_REQUEST = 100

# ----- SINGLE FILE ENDPOINT -----
@app.post("/metadata")
async def get_metadata():
    path = r"C:\Users\ZS932QF\Downloads\K76598.pdf"  # hardcoded
    try:
        metadata = extract_metadata(path)  # Process the file
        return metadata
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ----- MULTIPLE FILES ENDPOINT WITH LIMIT -----
class FileList(BaseModel):
    file_paths: List[str]

@app.post("/bulk-metadata")
async def get_bulk_metadata(files: FileList):
    if len(files.file_paths) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=413,  # Payload Too Large
            detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed per request."
        )
    tasks = [get_file_metadata_from_blob(path) for path in files.file_paths]
    results = await asyncio.gather(*tasks)
    return dict(zip(files.file_paths, results))

# ----- LOCAL DIRECTORY ENDPOINT -----
class Directory(BaseModel):
    directory_path: str

@app.post("/metadata-from-local-folder")
async def get_metadata_from_local_folder(directory: Directory):
    try:
        files = get_files_from_local_directory(directory.directory_path)

        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(
                status_code=413,
                detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed per request. Found {len(files)} files."
            )

        tasks = [get_file_metadata_from_blob(file) for file in files]
        results = await asyncio.gather(*tasks)

        # Ensure file paths are strings, not dicts
        return {file: metadata for file, metadata in zip(files, results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ----- AZURE CONTAINER ENDPOINT -----
@app.post("/metadata-from-azure-container")
async def get_metadata_from_azure_container():
    try:
        files = get_files_from_azure_blob_container()
        if len(files) > MAX_FILES_PER_REQUEST:
            raise HTTPException(
                status_code=413,
                detail=f"Maximum of {MAX_FILES_PER_REQUEST} files allowed per request. Found {len(files)} files."
            )

        tasks = [get_file_metadata_from_blob(file) for file in files]
        results = await asyncio.gather(*tasks)
        return dict(zip(files, results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ----- Dashboard Endpoints -------

# Database connection URL (replace with your actual database URL)
DATABASE_URL = "postgresql://postgres:postgres@localhost/pu2pay_master"

# Setup SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# SQLAlchemy model for Spend Data
class SpendData(Base):
    # __tablename__ = 'spend_data'
    __tablename__ = 'invoices'

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date)
    voucher_name = Column(String)
    branch = Column(String)
    currency = Column(String)
    # exchange_rate = Column(String)  # Keeping as String since your schema is text
    party = Column(String)
    quantity = Column(Numeric(20, 2), nullable=True)  # Keeping as String since your schema is text
    gross_amt = Column(Numeric(20, 2), nullable=True)
    discount = Column(Numeric(20, 2), nullable=True)
    gross_minus_discount = Column(Numeric(20, 2), nullable=True)
    # cgst = Column(String)
    # sgst = Column(String)
    # igst = Column(String)
    # cess = Column(String)
    # customs = Column(String)
    # total_customs_amount = Column(String)
    # gst_amount = Column(String)
    # rcmcgst = Column(String)
    # rcmsgst = Column(String)
    # rcmigst = Column(String)
    # rcmcess = Column(String)
    # rcm_amount = Column(String)
    net_amount = Column(Numeric(20, 2), nullable=True)
    # net_amount_in_company_currency = Column(String)
    # charges = Column(String)
    deductions = Column(Numeric(20, 2), nullable=True)
    # taxable_other_charges_amount = Column(String)
    # taxable_other_charges_tax_amount = Column(String)
    # round_off_amount = Column(String)
    total_value = Column(Numeric(20, 2), nullable=True)
    # total_value_in_company_currency = Column(String)
    # other_cost = Column(String)
    # other_cost_in_company_currency = Column(String)
    # quarter = Column(String)

# Create tables (this is done once when the app is first run)
Base.metadata.create_all(bind=engine)

# Pydantic models for request and response validation
class SpendDataResponse1(BaseModel):
    id: int
    date: str
    voucher_no: str
    branch: str
    currency: str
    exchange_rate: str
    party: str
    quantity: str
    gross_amount: str
    discount: str
    gross_minus_discount: str
    cgst: str
    sgst: str
    igst: str
    cess: str
    customs: str
    total_customs_amount: str
    gst_amount: str
    rcmcgst: str
    rcmsgst: str
    rcmigst: str
    rcmcess: str
    rcm_amount: str
    net_amount: str
    net_amount_in_company_currency: str
    charges: str
    deductions: str
    taxable_other_charges_amount: str
    taxable_other_charges_tax_amount: str
    round_off_amount: str
    total_value: str
    total_value_in_company_currency: str
    other_cost: str
    other_cost_in_company_currency: str
    quarter: str

    class Config:
        # orm_mode = True  # To make it compatible with SQLAlchemy models
        from_attributes = True

class SpendDataResponse(BaseModel):
    id: int
    date: date
    voucher_name: str
    branch: str
    currency: str
    party: str
    quantity: Optional[Decimal] = None
    gross_amount: Optional[Decimal] = None
    discount: Optional[Decimal] = None
    gross_minus_discount: Optional[Decimal] = None
    net_amount: Optional[Decimal] = None
    deductions: Optional[Decimal] = None
    total_value: Optional[Decimal] = None

#     id: Number,
#   date: Date,
#   voucher_name: string,
#   branch: string,
#   currency: string,
#   party: string,
#   quantity: Number,
#   gross_amt: Number,
#   discount: Number,
#   gross_minus_discount: Number,
#   net_amount: Number,
#   deductions: Number,
#   total_value: Number

    class Config:
        # orm_mode = True  # To make it compatible with SQLAlchemy models
        from_attributes = True

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_db_connection():
    try:
        # Create a session to interact with the database
        db = SessionLocal()
        
        # Run a simple query to check if the connection works
        result = db.execute(text("SELECT 1")).fetchone()
        
        if result:
            print("Database connection successful!")
        else:
            print("Database connection failed.")
        
        db.close()  # Always close the session when done
    except Exception as e:
        print(f"Error connecting to the database: {e}")

# Create a FastAPI app instance
app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Startup tasks (e.g., testing DB connection)
        test_db_connection()
        yield
    finally:
        # Cleanup tasks (e.g., close DB connections if needed)
        pass

app = FastAPI(lifespan=lifespan)

def test_db_connection():
    try:
        # Create a session to interact with the database
        db = SessionLocal()
        
        # Run a simple query to check if the connection works
        result = db.execute(text("SELECT 1")).fetchone()
        
        if result:
            print("Database connection successful!")
        else:
            print("Database connection failed.")
        
        db.close()  # Always close the session when done
    except Exception as e:
        print(f"Error connecting to the database: {e}")

@app.get("/")
async def root():
    return {"message": "Database connection check complete."}


# ----- Total Spend Breakdown by Category (Branch, Party, or Currency) -------
@app.get("/spend-data/branch/{branch}", response_model=List[SpendDataResponse])
def get_spend_data_by_branch(branch: str, db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).filter(SpendData.branch == branch).all()
    if not spend_data:
        raise HTTPException(status_code=404, detail="Branch not found")
    return spend_data

# ---------------- Spend Trend Over Time (By Quarter, Date) -----------------
@app.get("/spend-data/quarter/{quarter}", response_model=List[SpendDataResponse])
def get_spend_data_by_quarter(quarter: str, db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).filter(SpendData.quarter == quarter).all()
    if not spend_data:
        raise HTTPException(status_code=404, detail="Quarter not found")
    return spend_data

# ---------------- Cost Breakdown by Tax and Additional Charges -----------------
@app.get("/spend-data/taxes", response_model=List[SpendDataResponse])
def get_spend_data_by_taxes(db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).all()
    if not spend_data:
        raise HTTPException(status_code=404, detail="No data found")
    return spend_data

# ---------------- Spend vs. Discount and Gross Amount -----------------
@app.get("/spend-data", response_model=List[SpendDataResponse])
def get_spend_data(db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).all()
    print(f"Fetched data: {spend_data}")
    return spend_data

# ---------------- Cost per Party -----------------
@app.get("/spend-data/party/{party}", response_model=List[SpendDataResponse])
def get_spend_data_by_party(party: str, db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).filter(SpendData.party == party).all()
    return spend_data

# ---------------- Spend by Currency Exchange Rates -----------------
@app.get("/spend-data/currency/{currency}", response_model=List[SpendDataResponse])
def get_spend_data_by_currency(currency: str, db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).filter(SpendData.currency == currency).all()
    return spend_data

# ---------------- Spend by Quarter -----------------
@app.get("/spend-data/quarter/{quarter}", response_model=List[SpendDataResponse])
def get_spend_data_by_quarter(quarter: str, db: Session = Depends(get_db)):
    spend_data = db.query(SpendData).filter(SpendData.quarter == quarter).all()
    return spend_data

@app.get("/quantitybyBranch")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select branch, sum(quantity) as quantity from public.invoices group by branch")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalValuebyDate")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select date, sum(total_value) as total_value from public.invoices group by date")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/grossAmtwithDiscount")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select gross_amt, discount from public.invoices where discount is not null")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalValuebyParty")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select party, sum(total_value) as total_value from public.invoices group by party")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalValuebyYrQTR")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, EXTRACT(quarter from date) as qtr, sum(total_value) as total_value from public.invoices group by yr, qtr order by yr, qtr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalQtybyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(quantity) as quantity from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalPartiesbyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, count(distinct(party)) as parties from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalGrossSalesbyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(gross_amt) as gross_Sales from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalNetSalesbyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(net_amount) as net_Sales from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalInvoicesbyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, count(id) as invoices from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalTotalValueInvoicesbyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(total_value) as total_value from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/totalTotalDiscountbyYr")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(discount) as discount from public.invoices group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/top5/vendors/value")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select party as vendor, sum(total_value) as tot from public.invoices where total_value is not null group by party order by tot desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/top5/vendors/invoice")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select party as vendor, count(id) as invoices_cnt from public.invoices group by party order by invoices_cnt desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/top5/vendors/violation")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select vendor, sum(rejections) as rejections from public.reconciliation where rejections > 0 group by vendor order by rejections desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/top5/rejection/reasons")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select rejection_reason, sum(rejections) as rejections from public.reconciliation where rejections > 0 and rejection_reason is not null group by rejection_reason order by rejections desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/top5/vendor/item_mismatch")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select vendor, sum(item_mismatch) as item_mismatch from public.reconciliation where item_mismatch > 0 group by vendor order by item_mismatch desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/total/violatingVendors")
def get_active_users(db: Session = Depends(get_db)):
    # sql = text("select extract(year from date) as yr, vendor, sum(rejections) as rejections from public.reconciliation where rejections > 0 group by yr, vendor order by yr desc")
    sql = text("SELECT EXTRACT(YEAR FROM date) AS yr, COUNT(DISTINCT vendor) AS vendor_count FROM public.reconciliation WHERE rejections > 0 GROUP BY yr ORDER BY yr DESC")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/total/fullMatches")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(full_matches)as fullMatches from public.reconciliation group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/total/partialMatches")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(partial_matches)as partialMatches from public.reconciliation group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/total/rejections")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(rejections)as rejections from public.reconciliation group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/total/checkedInvoices")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select extract(year from date) as yr, sum(invoices)as checked_invoices from public.reconciliation group by yr")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/partialMatchesByDate")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select date, sum(partial_matches) as partial_matches from public.reconciliation group by date")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/fullMatchesByDate")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select date, sum(full_matches) as full_matches from public.reconciliation group by date")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/mismatch/currency")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select currency_mismatch as currency, sum(rejections) as rejections from public.reconciliation where rejections > 0 and currency_mismatch is not null group by currency order by rejections desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

@app.get("/tax_discrepancy/rejection_reason")
def get_active_users(db: Session = Depends(get_db)):
    sql = text("select rejection_reason, sum(tax_discrepancy) as tax_discrepancy from public.reconciliation where tax_discrepancy > 0 and rejection_reason is not null group by rejection_reason order by tax_discrepancy desc limit 5")
    result = db.execute(sql)
    data = result.mappings().all()  # Get list of dicts
    
    return data

# Update with your frontend's domain or use "*" for all origins (not recommended in production)
origins = [
    "http://localhost:4200",  # example for React local dev
    "http://127.0.0.1:3000",
    "https://your-frontend-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],             # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],             # Authorization, Content-Type, etc.
)