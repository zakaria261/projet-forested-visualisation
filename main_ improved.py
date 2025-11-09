from app_core_improved import app 
import app_callbacks_improved

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)