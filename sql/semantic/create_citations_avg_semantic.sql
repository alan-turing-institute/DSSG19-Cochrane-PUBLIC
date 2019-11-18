--copy over to clean
create table if not exists semantic.citations_avg (i int);
truncate semantic.citations_avg;
drop table if exists semantic.citations_avg;

--merge in paper id into papers_rgs_wide
create temporary table tbl as (select b.paperid, a.* from semantic.papers_rgs_wide a
left join semantic.recordid_paperid b on a.recordid=b.recordid);

--merge in labels into citations based on refpaperid
create temporary table tbl2 as (select a.paperid as c_pid, a.refpaperid, b.* from semantic.citations a
left join tbl b on a.refpaperid=b.paperid);

--merge in recordid for referring paperid
create temporary table tbl3 as (select b.recordid as c_recordid, a.* from tbl2 a
left join semantic.recordid_paperid b on a.c_pid=b.paperid);

--do large scale average
create temporary table tbl4 as (select
c_recordid,
avg(coalesce(sti::int,0)) as cited_sti,
avg(coalesce(childca::int,0)) as cited_childca,
avg(coalesce(neonatal::int,0)) as cited_neonatal,
avg(coalesce(back::int,0)) as cited_back,
avg(coalesce(depressn::int,0)) as cited_depressn,
avg(coalesce(movement::int,0)) as cited_movement,
avg(coalesce(preg::int,0)) as cited_preg,
avg(coalesce(menstr::int,0)) as cited_menstr,
avg(coalesce(muskel::int,0)) as cited_muskel,
avg(coalesce(epilepsy::int,0)) as cited_epilepsy,
avg(coalesce(lungca::int,0)) as cited_lungca,
avg(coalesce(infectn::int,0)) as cited_infectn,
avg(coalesce(haematol::int,0)) as cited_haematol,
avg(coalesce(dementia::int,0)) as cited_dementia,
avg(coalesce(muskinj::int,0)) as cited_muskinj,
avg(coalesce(inj::int,0)) as cited_inj,
avg(coalesce(pvd::int,0)) as cited_pvd,
avg(coalesce(sympt::int,0)) as cited_sympt,
avg(coalesce(pubhlth::int,0)) as cited_pubhlth,
avg(coalesce(liver::int,0)) as cited_liver,
avg(coalesce(oral::int,0)) as cited_oral,
avg(coalesce(eyes::int,0)) as cited_eyes,
avg(coalesce(addictn::int,0)) as cited_addictn,
avg(coalesce(compmed::int,0)) as cited_compmed,
avg(coalesce(prostate::int,0)) as cited_prostate,
avg(coalesce(skin::int,0)) as cited_skin,
avg(coalesce(stroke::int,0)) as cited_stroke,
avg(coalesce(uppergi::int,0)) as cited_uppergi,
avg(coalesce(coloca::int,0)) as cited_coloca,
avg(coalesce(wounds::int,0)) as cited_wounds,
avg(coalesce(airways::int,0)) as cited_airways,
avg(coalesce(occhealth::int,0)) as cited_occhealth,
avg(coalesce(tobacco::int,0)) as cited_tobacco,
avg(coalesce(endoc::int,0)) as cited_endoc,
avg(coalesce(vasc::int,0)) as cited_vasc,
avg(coalesce(ent::int,0)) as cited_ent,
avg(coalesce(cf::int,0)) as cited_cf,
avg(coalesce(ms::int,0)) as cited_ms,
avg(coalesce(fertilreg::int,0)) as cited_fertilreg,
avg(coalesce(neuromusc::int,0)) as cited_neuromusc,
avg(coalesce(renal::int,0)) as cited_renal,
avg(coalesce(epoc::int,0)) as cited_epoc,
avg(coalesce(commun::int,0)) as cited_commun,
avg(coalesce(gynaeca::int,0)) as cited_gynaeca,
avg(coalesce(ari::int,0)) as cited_ari,
avg(coalesce(breastca::int,0)) as cited_breastca,
avg(coalesce(emerg::int,0)) as cited_emerg,
avg(coalesce(schiz::int,0)) as cited_schiz,
avg(coalesce(hiv::int,0)) as cited_hiv,
avg(coalesce(behav::int,0)) as cited_behav,
avg(coalesce(htn::int,0)) as cited_htn,
avg(coalesce(ibd::int,0)) as cited_ibd,
avg(coalesce(incont::int,0)) as cited_incont,
avg(coalesce(anaesth::int,0)) as cited_anaesth,
1.0::float as citations_available
from tbl3 group by c_recordid);

create table semantic.citations_avg as table tbl4;
