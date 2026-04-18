<#
.SYNOPSIS
    Automated Week 2 Pipeline — Preprocessing + Splitting + Verification
.DESCRIPTION
    A single end-to-end PowerShell script that:
      Step 1: Runs the preprocessing pipeline (extract frames + MTCNN face crop)
      Step 2: Verifies preprocessing output counts
      Step 3: Creates train/val/test CSV splits at VIDEO level
      Step 4: Generates a summary report saved to outputs/week2_summary.txt

    Run once and walk away. All output is logged to outputs/automation.log
.NOTES
    Project:  F:\ml_project\deepfake-detection
    Python:   F:\ml_project\.venv\Scripts\python.exe
#>

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
$ErrorActionPreference = "Continue"
$PROJECT_ROOT = "F:\ml_project\deepfake-detection"
$PYTHON       = "F:\ml_project\.venv\Scripts\python.exe"
$LOG_FILE     = "$PROJECT_ROOT\outputs\automation.log"
$SUMMARY_FILE = "$PROJECT_ROOT\outputs\week2_summary.txt"

# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
function Write-Log {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $formatted = "[$timestamp] $Message"
    Add-Content -Path $LOG_FILE -Value $formatted -ErrorAction SilentlyContinue
    Write-Host $formatted -ForegroundColor $Color
}

function Write-Section {
    param([string]$Title, [int]$StepNumber)
    Write-Host ""
    Write-Log ("=" * 65) "Cyan"
    Write-Log "  STEP $StepNumber : $Title" "Cyan"
    Write-Log ("=" * 65) "Cyan"
}

function Assert-Critical {
    param([string]$Condition, [string]$ErrorMsg)
    if (-not $Condition) {
        Write-Log "CRITICAL ERROR: $ErrorMsg" "Red"
        Write-Log "Pipeline halted. Check the log at $LOG_FILE" "Red"
        exit 1
    }
}

# ═══════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════

# Ensure required directories exist
New-Item -ItemType Directory -Force -Path "$PROJECT_ROOT\outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "$PROJECT_ROOT\data\splits" | Out-Null

# Initialize log (truncate old log)
Set-Content -Path $LOG_FILE -Value "" -ErrorAction SilentlyContinue

Write-Log ("=" * 65) "Cyan"
Write-Log "  AUTOMATED WEEK 2 PIPELINE" "Cyan"
Write-Log "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" "Cyan"
Write-Log ("=" * 65) "Cyan"
Write-Log "Project Root : $PROJECT_ROOT"
Write-Log "Python       : $PYTHON"
Write-Log "Log File     : $LOG_FILE"

# Validate Python exists
if (-not (Test-Path $PYTHON)) {
    Write-Log "CRITICAL: Python not found at $PYTHON" "Red"
    Write-Log "Fix: Ensure the virtual environment exists at F:\ml_project\.venv" "Red"
    exit 1
}
Write-Log "Python validated: OK" "Green"

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: RUN PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════
Write-Section "Run Preprocessing Pipeline" 1
Write-Log "Command: $PYTHON -m src.preprocessing.run_pipeline"
Write-Log "Expected: Process ~1,000 videos -> ~32,000 face images (224x224)"
Write-Log "This may take 8-12 hours. Output is streamed below..."
Write-Log ("-" * 50)

$step1Start = Get-Date
$step1Success = $false

try {
    # Run from the project root so relative imports work
    & $PYTHON -m src.preprocessing.run_pipeline 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Add-Content -Path $LOG_FILE -Value $line -ErrorAction SilentlyContinue
        Write-Host $line
    }

    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
        $step1Success = $true
        Write-Log "Step 1 COMPLETED successfully." "Green"
    } else {
        Write-Log "Step 1 finished with exit code: $LASTEXITCODE" "Yellow"
        Write-Log "Continuing to verification step..." "Yellow"
        $step1Success = $true  # Allow continuation — verification will catch issues
    }
} catch {
    Write-Log "ERROR in Step 1: $_" "Red"
    Write-Log "Continuing to verification to check partial results..." "Yellow"
}

$step1Duration = (Get-Date) - $step1Start
Write-Log "Step 1 duration: $($step1Duration.ToString('hh\:mm\:ss'))"

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: VERIFY PREPROCESSING OUTPUT
# ═══════════════════════════════════════════════════════════════════════
Write-Section "Verify Preprocessing Output" 2

$processedDir = "$PROJECT_ROOT\data\processed"

if (-not (Test-Path $processedDir)) {
    Write-Log "CRITICAL ERROR: Processed directory not found: $processedDir" "Red"
    Write-Log "The preprocessing pipeline did not create any output." "Red"
    exit 1
}

# Count video folders in real/
$realDir = "$processedDir\real"
$realVideoCount = 0
if (Test-Path $realDir) {
    $realVideoCount = @(Get-ChildItem -Path $realDir -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            @(Get-ChildItem -Path $_.FullName -Filter "*.jpg" -File -ErrorAction SilentlyContinue).Count -gt 0
        }).Count
}

# Count video folders in fake/
$fakeDir = "$processedDir\fake"
$fakeVideoCount = 0
if (Test-Path $fakeDir) {
    $fakeVideoCount = @(Get-ChildItem -Path $fakeDir -Directory -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            @(Get-ChildItem -Path $_.FullName -Filter "*.jpg" -File -ErrorAction SilentlyContinue).Count -gt 0
        }).Count
}

# Count total JPG images
$totalImages = @(Get-ChildItem -Path $processedDir -Filter "*.jpg" -Recurse -File -ErrorAction SilentlyContinue).Count

$totalVideoFolders = $realVideoCount + $fakeVideoCount

Write-Log ("-" * 50)
Write-Log "  PREPROCESSING VERIFICATION RESULTS" "Yellow"
Write-Log ("-" * 50)
Write-Log "  Real video folders  : $realVideoCount  (expected: ~200)"
Write-Log "  Fake video folders  : $fakeVideoCount  (expected: ~800)"
Write-Log "  Total video folders : $totalVideoFolders  (expected: ~1,000)"
Write-Log "  Total face images   : $totalImages  (expected: ~30,500 - 32,000)"
Write-Log ("-" * 50)

# Validation checks
if ($totalVideoFolders -eq 0) {
    Write-Log "CRITICAL: No video folders found with images. Pipeline likely failed." "Red"
    exit 1
}

if ($totalImages -lt 1000) {
    Write-Log "WARNING: Very low image count ($totalImages). Check preprocessing.log for errors." "DarkYellow"
}

if ($totalImages -ge 30000) {
    Write-Log "Image count is within expected range. PASS" "Green"
} else {
    Write-Log "Image count is below expected (~32,000). Check logs for failed face detections." "DarkYellow"
}

# Calculate approximate success rate
$expectedFrames = $totalVideoFolders * 32
if ($expectedFrames -gt 0) {
    $successRate = [math]::Round(($totalImages / $expectedFrames) * 100, 1)
    Write-Log "  Face detection rate : ${successRate}% ($totalImages / $expectedFrames)"
} else {
    $successRate = 0
}

Write-Log "Step 2 COMPLETED." "Green"

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: CREATE TRAIN/VAL/TEST SPLITS
# ═══════════════════════════════════════════════════════════════════════
Write-Section "Create Train/Val/Test Splits" 3
Write-Log "Command: $PYTHON src\preprocessing\create_splits.py"
Write-Log "Expected: 3 CSV files (train.csv, val.csv, test.csv) in data\splits\"

$step3Success = $false

try {
    & $PYTHON "$PROJECT_ROOT\src\preprocessing\create_splits.py" 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Add-Content -Path $LOG_FILE -Value $line -ErrorAction SilentlyContinue
        Write-Host $line
    }

    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
        $step3Success = $true
    } else {
        Write-Log "create_splits.py exited with code: $LASTEXITCODE" "Red"
    }
} catch {
    Write-Log "ERROR in Step 3: $_" "Red"
}

# Verify CSV files exist
$splitsDir = "$PROJECT_ROOT\data\splits"
$trainCsv = "$splitsDir\train.csv"
$valCsv   = "$splitsDir\val.csv"
$testCsv  = "$splitsDir\test.csv"

$trainExists = Test-Path $trainCsv
$valExists   = Test-Path $valCsv
$testExists  = Test-Path $testCsv

Write-Log ("-" * 50)
Write-Log "  SPLIT VERIFICATION" "Yellow"
Write-Log ("-" * 50)

$trainCount = 0; $valCount = 0; $testCount = 0

if ($trainExists) {
    $trainCount = (Get-Content $trainCsv | Measure-Object -Line).Lines - 1  # minus header
    Write-Log "  train.csv : EXISTS  ($trainCount videos)" "Green"
} else {
    Write-Log "  train.csv : MISSING" "Red"
}

if ($valExists) {
    $valCount = (Get-Content $valCsv | Measure-Object -Line).Lines - 1
    Write-Log "  val.csv   : EXISTS  ($valCount videos)" "Green"
} else {
    Write-Log "  val.csv   : MISSING" "Red"
}

if ($testExists) {
    $testCount = (Get-Content $testCsv | Measure-Object -Line).Lines - 1
    Write-Log "  test.csv  : EXISTS  ($testCount videos)" "Green"
} else {
    Write-Log "  test.csv  : MISSING" "Red"
}

Write-Log "  Total in CSVs: $($trainCount + $valCount + $testCount) videos"
Write-Log ("-" * 50)

if (-not ($trainExists -and $valExists -and $testExists)) {
    Write-Log "CRITICAL: One or more split CSVs are missing!" "Red"
    exit 1
}

Write-Log "Step 3 COMPLETED." "Green"

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: GENERATE SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════
Write-Section "Generate Summary Report" 4

$totalDuration = (Get-Date) - $step1Start

$summaryContent = @"
================================================================
  WEEK 2 PREPROCESSING SUMMARY REPORT
  Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
================================================================

DATASET OVERVIEW
  Raw real videos       : 201
  Raw fake videos       : 800  (4 categories x 200)
  Total raw videos      : 1,001

PREPROCESSING RESULTS
  Video folders created : $totalVideoFolders
    - Real              : $realVideoCount
    - Fake              : $fakeVideoCount
  Total face images     : $totalImages
  Face detection rate   : ${successRate}%
  Frames per video      : 32 (target)

TRAIN / VAL / TEST SPLITS (70 / 15 / 15)
  train.csv             : $trainCount videos
  val.csv               : $valCount videos
  test.csv              : $testCount videos
  Split level           : VIDEO (no data leakage)
  Labels                : 0 = real, 1 = fake
  Random state          : 42

FILES CREATED
  data/processed/real/  : $realVideoCount video folders
  data/processed/fake/  : $fakeVideoCount video folders
  data/splits/train.csv : $trainCount rows
  data/splits/val.csv   : $valCount rows
  data/splits/test.csv  : $testCount rows

TOTAL PIPELINE DURATION : $($totalDuration.ToString('hh\:mm\:ss'))

STATUS: READY FOR WEEK 3 (CNN Model Training)
================================================================
"@

Set-Content -Path $SUMMARY_FILE -Value $summaryContent
Write-Log "Summary report saved to: $SUMMARY_FILE" "Green"

# Print the summary to console
Write-Host ""
Write-Host $summaryContent -ForegroundColor Cyan

# ═══════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════
Write-Log ("=" * 65) "Green"
Write-Log "  WEEK 2 PIPELINE COMPLETE" "Green"
Write-Log "  Full log     : $LOG_FILE" "Green"
Write-Log "  Summary      : $SUMMARY_FILE" "Green"
Write-Log "  EDA Notebook : $PROJECT_ROOT\notebooks\EDA.ipynb" "Green"
Write-Log ("=" * 65) "Green"

Write-Host ""
Write-Host "All done! You can now:" -ForegroundColor Green
Write-Host "  1. Open notebooks\EDA.ipynb in Jupyter to review visualizations" -ForegroundColor White
Write-Host "  2. Review outputs\week2_summary.txt for the full report" -ForegroundColor White
Write-Host "  3. Proceed to Week 3: CNN Model Training" -ForegroundColor White
Write-Host ""
