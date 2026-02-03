import redis
import sys

def check_redis():
    print("Checking Redis connection at localhost:6379...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=3.0)
        r.ping()
        print("Redis is UP and reachable.")
        return True
    except redis.exceptions.ConnectionError:
        print("Redis is DOWN or Unreachable.")
        return False
    except Exception as e:
        print(f"Error checking Redis: {e}")
        return False

if __name__ == "__main__":
    if not check_redis():
        sys.exit(1)
