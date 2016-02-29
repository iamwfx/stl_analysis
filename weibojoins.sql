Drop table cd_residentialgrids_weibocount;
Create table cd_residentialgrids_weibocount as (
	select cd_residentialgrids.gid, 
	cd_residentialgrids.geom
	from cd_residentialgrids

);

Alter table cd_residentialgrids_weibocount 
	add column counts_unique integer default 0,
	add column counts_total integer default 0
	;


Alter table chengdu_poi_20151213
--	drop column resi_gid,
	add column resi_gid integer default 0;
	
-- Change projection in order make the join
select UpdateGeometrySRID('chengdu_poi_20151213','geom',4326);

-- Join each residential grid to a POI point, in order preserve the value of checkins
Update chengdu_poi_20151213
	set resi_gid = f.resi_gid
	From(
		select r.gid as resi_gid,
		w.gid 
--		From cd_residentialgrids r 
--		left outer join cd_weibo_poi_points w
		From  chengdu_poi_20151213 w
		left outer join cd_residentialgrids r
		on st_intersects(ST_Transform(w.geom,4326),ST_Transform(r.geom,4326))
	--	r.geom,w.geom) 
		) as f
	where f.gid = chengdu_poi_20151213.gid;

-- Aggregate total counts into a new table
Drop table cd_residentialgrids_weibocount;
Create table cd_residentialgrids_weibocount as 
	select r.gid, sum(p.checkin_nu) as count
	from chengdu_poi_20151213 as p
	right outer join cd_residentialgrids r
	on p.resi_gid = r.gid
	group by r.gid
	; 

-- Replace nulls
select COALESCE( NULLIF(cd_residentialgrids_weibocount.count,Null) , '0' ) from 

select UpdateGeometrySRID('public','cd_weibo_poi_points','the_geom',4326);
select UpdateGeometrySRID('public','cd_residentialgrids','geom',4326);
select find_srid('stl','cd_weibo_poi_points','the_geom');
