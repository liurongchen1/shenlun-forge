#!/usr/bin/env python3
"""
ShenlunForge Quick Demo
=======================

Demonstrates the 4-step compression pipeline on a sample conversation.
"""

import asyncio
import sys
import os

# Add parent dir to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shenlun_forge import ShenlunForge, TaskType


# Sample conversation that would benefit from compression
SAMPLE_CONVERSATION = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Hi! I'm building a web app for my restaurant. I need help with the backend."},
    {"role": "assistant", "content": "Hello! I'd be happy to help you build a backend for your restaurant web app. What features do you need?"},
    {"role": "user", "content": "I need a menu management system. We have about 50 dishes across 5 categories: appetizers, mains, desserts, drinks, and specials."},
    {"role": "assistant", "content": "Great! Let's design the database schema. For 50 dishes across 5 categories, I'd suggest a PostgreSQL database with a 'categories' table and a 'dishes' table with a foreign key to categories. Each dish would have: name, description, price, category_id, is_available, and created_at fields."},
    {"role": "user", "content": "Sounds good. I also need user authentication. We have staff and admin roles."},
    {"role": "assistant", "content": "For authentication, I recommend using JWT tokens. You'll need a 'users' table with fields: id, username, password_hash, role (staff/admin), and last_login. I suggest using bcrypt for password hashing and PyJWT for token management."},
    {"role": "user", "content": "Can you show me the code for the database models? I'm using Python with SQLAlchemy."},
    {"role": "assistant", "content": "```python\nfrom sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime\nfrom sqlalchemy.orm import relationship, declarative_base\nfrom datetime import datetime\n\nBase = declarative_base()\n\nclass Category(Base):\n    __tablename__ = 'categories'\n    id = Column(Integer, primary_key=True)\n    name = Column(String(50), nullable=False)\n    dishes = relationship('Dish', back_populates='category')\n\nclass Dish(Base):\n    __tablename__ = 'dishes'\n    id = Column(Integer, primary_key=True)\n    name = Column(String(100), nullable=False)\n    description = Column(String(500))\n    price = Column(Float, nullable=False)\n    category_id = Column(Integer, ForeignKey('categories.id'))\n    is_available = Column(Boolean, default=True)\n    created_at = Column(DateTime, default=datetime.utcnow)\n    category = relationship('Category', back_populates='dishes')\n\nclass User(Base):\n    __tablename__ = 'users'\n    id = Column(Integer, primary_key=True)\n    username = Column(String(50), unique=True, nullable=False)\n    password_hash = Column(String(128), nullable=False)\n    role = Column(String(20), default='staff')\n    last_login = Column(DateTime)\n```"},
    {"role": "user", "content": "Now let's add the API endpoints. I need CRUD for dishes and categories."},
    {"role": "assistant", "content": "Here are the FastAPI endpoints:\n\n```python\nfrom fastapi import FastAPI, HTTPException, Depends\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\n# Dish endpoints\n@app.get('/api/dishes')\nasync def list_dishes(category: str = None, available_only: bool = False):\n    # Query logic here\n    pass\n\n@app.post('/api/dishes')\nasync def create_dish(dish: DishCreate):\n    # Create logic\n    pass\n\n@app.put('/api/dishes/{dish_id}')\nasync def update_dish(dish_id: int, dish: DishUpdate):\n    # Update logic\n    pass\n\n@app.delete('/api/dishes/{dish_id}')\nasync def delete_dish(dish_id: int):\n    # Delete logic\n    pass\n```"},
    {"role": "user", "content": "I need to add order management too. Orders have multiple dishes, quantities, and status tracking."},
    {"role": "assistant", "content": "For order management, you'll need:\n\n1. An 'orders' table: id, user_id, status (pending/confirmed/preparing/ready/completed/cancelled), total_amount, created_at, updated_at\n2. An 'order_items' table: id, order_id, dish_id, quantity, unit_price, subtotal\n\nThe order workflow: pending → confirmed → preparing → ready → completed. Each status change should trigger notifications."},
    {"role": "user", "content": "OK, let me implement this and test it. I'll get back to you if I run into issues."},
    {"role": "assistant", "content": "Sounds good! Let me know if you need help with anything. Some things to consider:\n- Add database migrations using Alembic\n- Add input validation for prices (must be positive)\n- Consider adding soft delete for dishes (is_deleted flag) instead of hard delete\n- Add rate limiting to your API endpoints\n- Consider adding a simple caching layer for the menu (dishes don't change often)"},
    {"role": "user", "content": "Great tips, thanks! I'll add those. By the way, what did we decide for the database and auth setup?"},
]


async def main():
    """Run the demo."""
    print("=" * 60)
    print("ShenlunForge Demo — ACS Context Compression")
    print("=" * 60)

    # Calculate original token count (rough estimate)
    original_text = " ".join(m["content"] for m in SAMPLE_CONVERSATION)
    original_tokens = len(original_text) // 3  # Rough estimate

    print(f"\n📝 Original conversation:")
    print(f"   Messages: {len(SAMPLE_CONVERSATION)}")
    print(f"   Estimated tokens: ~{original_tokens}")
    print(f"   Characters: {len(original_text)}")

    # 'simple' needs no API key and no network. Pass strategy="acs" together
    # with an API key to run the full LLM-driven pipeline instead.
    forge = ShenlunForge.create(strategy="simple")

    # Compress with the user's query
    current_query = "What did we decide for the database and auth setup?"
    print(f"\n🔍 Current query: \"{current_query}\"")
    print(f"\n⚙️  Running 4-step compression pipeline...")

    result = await forge.compress(
        messages=SAMPLE_CONVERSATION,
        current_query=current_query,
        max_tokens=800,
        verbose=True,
    )

    # Display results
    print(f"\n{'=' * 60}")
    print("COMPRESSION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n📊 Statistics:")
    print(f"   Original tokens:  ~{result.original_tokens}")
    print(f"   Compressed tokens: ~{result.compressed_tokens}")
    print(f"   Compression ratio: {result.compression_ratio:.1%}")
    print(f"   Blocks merged: {result.stats.get('compressed_groups', '?')} groups from {result.stats.get('relevant_blocks', '?')} relevant blocks")
    print(f"   LLM used: {result.stats.get('llm_available')}   degraded: {result.stats.get('degraded')}")
    if result.stats.get("passthrough"):
        print("   Note: nothing was gained, the original context was returned unchanged.")

    print(f"\nStep-by-step log:")
    for step_name, step_data in result.steps_log.items():
        print(f"   {step_name}: {step_data}")

    print(f"\n📄 Compressed context:")
    print(f"{'─' * 60}")
    print(result.compressed_text)
    print(f"{'─' * 60}")

    # Show block-level details
    print(f"\n🧩 Compressed blocks ({len(result.blocks)}):")
    for i, block in enumerate(result.blocks):
        print(f"   [{i}] label={block.summary_label or 'N/A'}, "
              f"type={block.info_type}, "
              f"ratio={block.compression_ratio:.2f}, "
              f"from messages {block.original_indices}")
        preview = block.content[:100] + "..." if len(block.content) > 100 else block.content
        print(f"       {preview}")


if __name__ == "__main__":
    asyncio.run(main())
