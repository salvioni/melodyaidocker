"""Typed Supabase schema models for Python.

Generated from the project's TypeScript schema definition.
"""

from __future__ import annotations

from typing import Literal, Required, TypedDict


Json = (
	str
	| int
	| float
	| bool
	| None
	| dict[str, "Json | None"]
	| list["Json"]
)

SongProvider = Literal["genius", "user_upload", "youtube", "spotify"]


class PlaylistItemsRow(TypedDict):
	created_at: str
	id: int
	playlist_id: int
	song_id: int


class PlaylistItemsInsert(TypedDict, total=False):
	created_at: str
	id: int
	playlist_id: Required[int]
	song_id: Required[int]


class PlaylistItemsUpdate(TypedDict, total=False):
	created_at: str
	id: int
	playlist_id: int
	song_id: int


class PlaylistsRow(TypedDict):
	created_at: str
	id: int
	is_public: bool
	title: str
	user_id: str


class PlaylistsInsert(TypedDict, total=False):
	created_at: str
	id: int
	is_public: bool
	title: Required[str]
	user_id: str


class PlaylistsUpdate(TypedDict, total=False):
	created_at: str
	id: int
	is_public: bool
	title: str
	user_id: str


class SongsRow(TypedDict):
	artist: str
	created_at: str
	id: int
	owner_id: str | None
	provider: SongProvider
	provider_id: str | None
	status: str
	title: str


class SongsInsert(TypedDict, total=False):
	artist: Required[str]
	created_at: str
	id: int
	owner_id: str | None
	provider: SongProvider
	provider_id: str | None
	status: str
	title: Required[str]


class SongsUpdate(TypedDict, total=False):
	artist: str
	created_at: str
	id: int
	owner_id: str | None
	provider: SongProvider
	provider_id: str | None
	status: str
	title: str


class TablesRows(TypedDict):
	playlist_items: PlaylistItemsRow
	playlists: PlaylistsRow
	songs: SongsRow


class TablesInsert(TypedDict):
	playlist_items: PlaylistItemsInsert
	playlists: PlaylistsInsert
	songs: SongsInsert


class TablesUpdate(TypedDict):
	playlist_items: PlaylistItemsUpdate
	playlists: PlaylistsUpdate
	songs: SongsUpdate


class PublicEnums(TypedDict):
	song_provider: SongProvider


class PublicSchema(TypedDict):
	Tables: TablesRows
	TablesInsert: TablesInsert
	TablesUpdate: TablesUpdate
	Enums: PublicEnums


class InternalSupabase(TypedDict):
	PostgrestVersion: Literal["14.1"]


class Database(TypedDict):
	__InternalSupabase: InternalSupabase
	public: PublicSchema


CONSTANTS = {
	"public": {
		"Enums": {
			"song_provider": ["genius", "user_upload", "youtube", "spotify"],
		},
	},
}

