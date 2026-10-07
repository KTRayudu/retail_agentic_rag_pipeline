from qdrant_client import QdrantClient

def check():
    try:
        client = QdrantClient("http://localhost:6333")
        collections = client.get_collections()
        
        print("\n" + "="*40)
        print("🔍 CURRENT QDRANT COLLECTIONS")
        print("="*40)
        
        if not collections.collections:
            print("No collections found in Qdrant!")
            return
            
        for c in collections.collections:
            info = client.get_collection(c.name)
            print(f"\n📁 Collection Name: {c.name}")
            print(f"   ↳ Number of items: {info.points_count}")
            print(f"   ↳ Status: {info.status.name}")
            
        print("\n" + "="*40 + "\n")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}\n(Make sure your docker-compose is running!)")

if __name__ == "__main__":
    check()
