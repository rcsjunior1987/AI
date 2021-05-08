SELECT plans.member_key
     , plans.plan_key 
     , plans.start_date plan_start_date
     , plans.end_date plan_start_date
     , plans.status
     , budget.id id_budget
     , budget.booking_number budget_number
     
<<<<<<< HEAD
	 , budget.opening_balance
     , budget.closing_balance	
         	
     , IFNULL(level2.key
            , level2_from_level3.key) level2_key   
        
     , IFNULL(level2.name
            , level2_from_level3.name) level2_name
            
     , IFNULL(level1_from_level2.key
            , level1_from_level3.key) level1_key
     
     , IFNULL(level1_from_level2.name
			, level1_from_level3.name) level1_name
          
=======
     , level3.item_category_level2_id,
     , budget.item_category_level2_key
     
     
     
     , IFNULL(level3.item_category_level2_id, budget.item_category_level2_key) item_category_level2_key
     , IFNULL(level2_from_level3.name, level2.name) level2_name
     , IFNULL(level2_from_level3.code, level2.code) level2_code
     , IFNULL(level2_from_level3.display_name, level2.display_name) level2_display_name
     , IFNULL(level2_from_level3.item_category_level1_id, level2.item_category_level1_id) level1_id
     , IFNULL(level1_from_level3.name, level1_from_level2.name) level1_name
>>>>>>> 8f720b2b4e9cec63e694a4d42aa154bd5f56dde0
     , budget.item_category_level3_key
     , item_ref_from_level3.name budget_level3_name
     
     , budget.allocation value_allocated_budget
     , budget.remaining value_remained_budget
     , budget.status status_budget     
  FROM HH_plan_budget budget
  
  INNER JOIN (
				SELECT plan.plan_key
				     , plan.member_key
					 , MAX(plan.start_date) start_date
					 , plan.end_date
				     , plan.status
				  FROM HH_plan plan
<<<<<<< HEAD
				 WHERE plan.status = 'COMPLETED'
=======
				 WHERE plan.status = 'COMPLETED'  
>>>>>>> 8f720b2b4e9cec63e694a4d42aa154bd5f56dde0
				  GROUP BY plan.member_key
  
				UNION

				SELECT plan.plan_key
				     , plan.member_key member_key
					 , MAX(plan.start_date) start_date
					 , plan.end_date
				     , plan.status  
				  FROM HH_plan plan
				 WHERE plan.status = 'PLAN_DELIVERY_ACTIVE'
				  GROUP BY plan.member_key
			 ) plans on budget.plan_key = plans.plan_key
  
	LEFT JOIN HH_item_category_level2 level2
					
            JOIN HH_item_category_level1 level1_from_level2
		  	  ON level1_from_level2.id = level2.item_category_level1_id
		   
           ON budget.item_category_level2_key IS NOT NULL
          AND level2.key = budget.item_category_level2_key
          
    LEFT JOIN HH_item_category_level3 level3
			
            JOIN HH_item_category_level2 level2_from_level3
              ON level2_from_level3.id = level3.item_category_level2_id
              
            JOIN HH_item_category_level1 level1_from_level3
              ON level1_from_level3.id = level2_from_level3.item_category_level1_id
              
            JOIN HH_ndis_service_item_ref item_ref_from_level3
              ON item_ref_from_level3.reference_number = level3.reference_number
           
           ON budget.item_category_level3_key IS NOT NULL
	      AND level3.key = budget.item_category_level3_key