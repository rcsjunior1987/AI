SELECT member.id
     , member.member_key
     , member.date_of_birth
     , member.first_name
     , member.last_name
     , member.price_zone_code
     , csm.u_disabilities
     , csm.u_gender
     , region.SA1
     , region.SA2
     , region.SA3
     , region.SA4
  FROM SNOW_csm_consumer_user csm
  
  INNER JOIN HH_member member
          ON member.membership_number = csm.u_ndis_number
          
  INNER JOIN libe_leapinprod_person person
          ON person.id = csm.u_leapin_id
          
  LEFT JOIN libe_leapinprod_memberregion region
        ON region.MemberId = member.id
          
 WHERE csm.u_stage = 'li_managed'
 
  ORDER BY member.first_name
         , member.last_name
         , member.date_of_birth