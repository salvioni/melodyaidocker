"""Server configuration from environment variables."""

import os

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://mrmtvuviquucqlvzwejt.supabase.co",
)
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1ybXR2dXZpcXV1Y3Fsdnp3ZWp0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2OTk2MTI3OCwiZXhwIjoyMDg1NTM3Mjc4fQ.59bZVaquEg2-M3RR8aA64ytggk6ttH5nq82hgSD1joQ",
)
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/melody")

# Signed URL expiry in seconds (3 hours)
SIGNED_URL_EXPIRY = 60 * 60 * 3

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCsD79GNInoCJ-cDYYBUVGTl_rBuo7uML8")
