 """  
Safe, untrusted dataset ingestion module for CSV and Excel files.
Supports multi-sheet Excel files, encoding fallbacks, and content validation.
"""

import io
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from src.security import validate_file_metadata


class DataLoader:
    """Handles secure loading, parsing, and inspection of tabular datasets."""

    @staticmethod
    def inspect_excel_sheets(file_bytes: bytes) -> List[str]:
        """List sheet names available in an Excel file without loading entire data."""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")
            return excel_file.sheet_names
        except Exception as e:
            raise ValueError(f"Failed to inspect Excel sheets: {str(e)}")

    @staticmethod
    def load_dataset(
        file_input: Union[io.BytesIO, bytes, str],
        filename: str,
        sheet_name: Optional[str] = None,
        file_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Safely loads a CSV or Excel dataset into a pandas DataFrame.
        Returns the DataFrame and a metadata summary dict.
        """
        if isinstance(file_input, bytes):
            actual_size = len(file_input)
            stream = io.BytesIO(file_input)
        elif hasattr(file_input, "read"):
            # Streamlit UploadedFile or file-like
            raw_bytes = file_input.read()
            actual_size = len(raw_bytes)
            stream = io.BytesIO(raw_bytes)
            # Reset pointer if needed
            if hasattr(file_input, "seek"):
                file_input.seek(0)
        else:
            raise TypeError("Expected bytes or file-like object.")

        if file_size is None:
            file_size = actual_size

        # Security check
        is_valid, msg = validate_file_metadata(filename, file_size)
        if not is_valid:
            raise ValueError(msg)

        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        df = None
        if ext == ".csv":
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            last_err = None
            
            # Step 1: Try default C engine across encodings
            for enc in encodings:
                try:
                    stream.seek(0)
                    df = pd.read_csv(stream, encoding=enc, low_memory=False)
                    break
                except UnicodeDecodeError as e:
                    last_err = e
                    continue
                except Exception as e:
                    last_err = e
                    break  # Not an encoding issue, proceed to robust fallback
            
            # Step 2: Fallback to Python engine if C-engine hits tokenizing/buffer overflow/unquoted multiline fields
            if df is None:
                for enc in encodings:
                    try:
                        stream.seek(0)
                        df = pd.read_csv(
                            stream,
                            encoding=enc,
                            engine="python",
                            on_bad_lines="skip",
                        )
                        break
                    except UnicodeDecodeError as e:
                        last_err = e
                        continue
                    except Exception as e:
                        last_err = e
                        continue

            if df is None:
                raise ValueError(f"Failed to parse CSV file: {str(last_err)}")

        elif ext in (".xlsx", ".xls"):
            try:
                stream.seek(0)
                engine = "openpyxl" if ext == ".xlsx" else None
                # If sheet_name is provided, use it; else first sheet
                target_sheet = sheet_name if sheet_name else 0
                df = pd.read_excel(stream, sheet_name=target_sheet, engine=engine)
            except Exception as e:
                raise ValueError(f"Failed to parse Excel file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        if df is None or df.empty:
            if df is not None and df.shape[1] > 0 and df.shape[0] == 0:
                # Schema exists but zero rows
                pass
            else:
                raise ValueError("Dataset is empty. No rows or columns found.")

        # Ensure column names are strings and strip leading/trailing whitespace from column headers
        df.columns = [str(c).strip() for c in df.columns]

        meta = {
            "filename": filename,
            "row_count": int(df.shape[0]),
            "col_count": int(df.shape[1]),
            "columns": list(df.columns),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
            "sheet_name": sheet_name,
        }

        return df, meta
